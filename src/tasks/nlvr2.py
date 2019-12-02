# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

from functools import partial
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from param import args
from tasks.nlvr2_model import NLVR2Model
from tasks.nlvr2_data import NLVR2Dataset, NLVR2TorchDataset, NLVR2Evaluator

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')
MixMatchDataTuple = collections.namedtuple("DataTuple", 'dataset loader unlabel_dataset unlabel_loader evaluator')


def get_tuple(splits: str, bs: int, shuffle=False, drop_last=False) -> DataTuple:
    dset = NLVR2Dataset(splits)
    tset = NLVR2TorchDataset(dset)
    evaluator = NLVR2Evaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )
    print("Final length of splits %s = %d" % (splits, len(tset)))

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


def shuffle_mixup_tensor(tensor1, mixup_parameters):
    random_indices = mixup_parameters["random_indices"]
    lambda_sample = mixup_parameters["lambda_sample"]

    shuffle_feats = tensor1[random_indices, :]
    mixup_feats = lambda_sample * tensor1 + (1.0 - lambda_sample) * shuffle_feats
    return mixup_feats


def mixup_tensor1_tensor2(tensor1, tensor2, mixup_parameters):
    lambda_sample = mixup_parameters["lambda_sample"]
    assert tensor1.shape == tensor2.shape
    assert tensor1.shape[0] == lambda_sample.shape[0]
    mixup_feats = lambda_sample * tensor1 + (1.0 - lambda_sample) * tensor2
    return mixup_feats


class NLVR2BaseClass:
    def __init__(self):
        pass

    def setup_model(self):
        self.model = NLVR2Model()
        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)

        # GPU options
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()
        self.model = self.model.cuda()

    def setup_optimizers(self):
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(list(self.model.parameters()), args.lr)

    def single_predict(self, datum_tuple):
        ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
        with torch.no_grad():
            feats, boxes = feats.cuda(), boxes.cuda()
            logit, _, _ = self.model(feats, boxes, sent)
            score, predict = logit.max(1)
            probs = torch.nn.functional.softmax(logit, dim=-1)
        return ques_id, score, predict, probs

    def prepare_valid_data(self):
        if args.valid != "":
            valid_bsize = 2048 if args.multiGPU else 512
            self.valid_tuple = get_tuple(
                args.valid, bs=valid_bsize,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None

    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, score, predict, _ = self.single_predict(datum_tuple)
            for qid, l in zip(ques_id, predict.cpu().numpy()):
                quesid2ans[qid] = l
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans = self.predict(eval_tuple, dump)
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


class NLVR2MixMatch(NLVR2BaseClass):
    def __init__(self):
        super(NLVR2MixMatch, self).__init__()
        self.prepare_train_data()
        self.prepare_valid_data()
        self.setup_model()
        self.setup_losses()
        self.setup_optimizers()

        self.beta_distro = torch.distributions.beta.Beta(args.beta_distro_alpha, args.beta_distro_alpha)

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def prepare_train_data(self):
        label_dset = NLVR2Dataset(splits=args.train,
                                  fraction=args.mixmatch_label_fraction)
        unlabel_dset = NLVR2Dataset(splits=args.train,
                                    fraction=args.mixmatch_label_fraction,
                                    remove_labels=True,
                                    keep_reverse=True)
        print("Final Labelled / Unlabelled Split = %d / %d" % (len(label_dset), len(unlabel_dset)))
        label_tset = NLVR2TorchDataset(label_dset)
        unlabel_tset = NLVR2TorchDataset(unlabel_dset)

        label_data_loader = DataLoader(
            label_tset, batch_size=args.batch_size // 2,
            shuffle=True, num_workers=args.num_workers,
            drop_last=True, pin_memory=True
        )
        unlabel_data_loader = DataLoader(
            unlabel_tset, batch_size=args.batch_size // 2,
            shuffle=True, num_workers=args.num_workers,
            drop_last=True, pin_memory=True
        )

        evaluator = NLVR2Evaluator(label_dset)

        self.train_tuple = MixMatchDataTuple(dataset=label_dset,
                                             loader=label_data_loader,
                                             unlabel_dataset=unlabel_dset,
                                             unlabel_loader=unlabel_data_loader,
                                             evaluator=evaluator)

    def setup_losses(self):
        self.mixmatch_loss = nn.KLDivLoss(reduction='batchmean')

    def sample_beta_max(self, sample_shape):
        lambda_sample = self.beta_distro.sample(sample_shape=sample_shape).cuda()
        lambda_sample = torch.max(lambda_sample, 1.0 - lambda_sample)
        return lambda_sample

    def train(self, train_tuple, eval_tuple):
        label_dataset, label_loader, unlabel_dataset, unlabel_loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(label_loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, ((label_ques_id, label_feats, label_boxes, label_sents, label),
                    (unlabel_ques_id, unlabel_feats, unlabel_boxes, unlabel_sents)) in iter_wrapper(enumerate(zip(label_loader, unlabel_loader))):
                self.optim.zero_grad()

                label_feats, label_boxes, label = label_feats.cuda(), label_boxes.cuda(), label.cuda()
                unlabel_feats, unlabel_boxes = unlabel_feats.cuda(), unlabel_boxes.cuda()

                # Begin MixMatch Algorithm
                # Step 1 - Find current predictions on the unlabeled batch
                self.model.eval()
                _, _, preds, probs = self.single_predict((unlabel_ques_id, unlabel_feats, unlabel_boxes, unlabel_sents))

                # Step 2 - Sharpen predictions. In this case, round them off
                labels_one_hot = torch.nn.functional.one_hot(label, num_classes=2)
                unlabel_preds_one_hot = torch.nn.functional.one_hot(preds, num_classes=2)

                # Step 3 - Concatenate the labeled and unlabeled examples
                all_feats = torch.cat([label_feats, unlabel_feats], dim=0)
                all_boxes = torch.cat([label_boxes, unlabel_boxes], dim=0)
                all_sents = label_sents + unlabel_sents
                all_labels = torch.cat([labels_one_hot, unlabel_preds_one_hot], dim=0)

                # Step 4 - MixUp labelled and unlabelled examples
                # Compute LXRT features for points in dataset
                self.model.train()

                _, _, all_lxrt_feats = self.model(all_feats, all_boxes, all_sents)

                # Shuffle the concatenated batch
                random_indices = torch.randperm(labels_one_hot.shape[0] + unlabel_preds_one_hot.shape[0])
                shuffled_all_lxrt_feats = all_lxrt_feats[random_indices, :]
                shuffled_all_labels = all_labels[random_indices, :]

                # Step 4.1 - Mixup on the labeled examples (Algo 1, Line 13)
                mixup_parameters = {
                    "lambda_sample": self.sample_beta_max((labels_one_hot.shape[0], 1))
                }
                mixup_t1_t2_fn = partial(mixup_tensor1_tensor2, mixup_parameters=mixup_parameters)

                mixmatch_labeled_labels = mixup_t1_t2_fn(tensor1=labels_one_hot,
                                                         tensor2=shuffled_all_labels[:labels_one_hot.shape[0]])
                _, mixmatch_labeled_logits, _ = self.model(feat=label_feats,
                                                           pos=label_boxes,
                                                           sent=label_sents,
                                                           mixup_tensor_fn=partial(mixup_t1_t2_fn,
                                                                                   tensor2=shuffled_all_lxrt_feats[:labels_one_hot.shape[0]]))
                mixmatch_labeled_log_softmax = torch.nn.functional.log_softmax(mixmatch_labeled_logits, dim=1)

                # Step 4.2 - Mixup on the unlabeled examples (Algo 1, Line 14)
                mixup_parameters = {
                    "lambda_sample": self.sample_beta_max((unlabel_preds_one_hot.shape[0], 1))
                }
                mixup_t1_t2_fn = partial(mixup_tensor1_tensor2, mixup_parameters=mixup_parameters)

                mixmatch_unlabeled_labels = mixup_t1_t2_fn(tensor1=unlabel_preds_one_hot,
                                                           tensor2=shuffled_all_labels[labels_one_hot.shape[0]:])
                _, mixmatch_unlabeled_logits, _ = self.model(feat=unlabel_feats,
                                                             pos=unlabel_boxes,
                                                             sent=unlabel_sents,
                                                             mixup_tensor_fn=partial(mixup_t1_t2_fn,
                                                                                     tensor2=shuffled_all_lxrt_feats[labels_one_hot.shape[0]:]))
                mixmatch_unlabeled_log_softmax = torch.nn.functional.log_softmax(mixmatch_unlabeled_logits, dim=1)

                # Compute losses on two mixed batches separately
                labeled_loss = self.mixmatch_loss(input=mixmatch_labeled_log_softmax,
                                                  target=mixmatch_labeled_labels)
                unlabeled_loss = self.mixmatch_loss(input=mixmatch_unlabeled_log_softmax,
                                                    target=mixmatch_unlabeled_labels)

                loss = labeled_loss + args.mixmatch_unlabel_weight * unlabeled_loss

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, predict = mixmatch_labeled_logits.max(1)
                for qid, l in zip(label_ques_id, predict.cpu().numpy()):
                    quesid2ans[qid] = l

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")


class NLVR2(NLVR2BaseClass):
    def __init__(self):
        super(NLVR2MixMatch, self).__init__()
        self.train_tuple = get_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        self.prepare_valid_data()

        if args.mixup:
            self.beta_distro = torch.distributions.beta.Beta(args.beta_distro_alpha, args.beta_distro_alpha)

        self.setup_model()
        self.setup_losses()
        self.setup_optimizers()

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def setup_losses(self):
        # Losses and optimizer
        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if args.mixup:
            self.mixup_loss = nn.KLDivLoss(reduction='batchmean')

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, label) in iter_wrapper(enumerate(loader)):
                self.model.train()
                self.optim.zero_grad()

                feats, boxes, label = feats.cuda(), boxes.cuda(), label.cuda()

                if args.mixup:
                    # Sample a shuffled batch and lambda values for the current batch
                    mixup_parameters = {
                        "random_indices": torch.randperm(label.shape[0]).cuda(),
                        "lambda_sample": self.beta_distro.sample(sample_shape=(label.shape[0], 1)).cuda()
                    }
                    # Freeze these parameters for the mixup
                    mixup_tensor_fn = partial(shuffle_mixup_tensor, mixup_parameters=mixup_parameters)
                    # Apply mixup on one-hot labels
                    one_hot_labels = torch.nn.functional.one_hot(label, num_classes=2)
                    mixup_labels = mixup_tensor_fn(tensor=one_hot_labels)
                    # Obtain usual logits and logits after mixing LXRT features
                    logit, mixed_logits, _ = self.model(feats, boxes, sent, mixup_tensor_fn)
                    # Usual CE loss
                    loss1 = self.mce_loss(logit, label)
                    # Loss with soft labels for mixup
                    mixed_log_softmax = torch.nn.functional.log_softmax(mixed_logits, dim=1)
                    loss2 = self.mixup_loss(input=mixed_log_softmax,
                                            target=mixup_labels)
                    loss = loss1 + loss2
                else:
                    logit, _, _ = self.model(feats, boxes, sent)
                    loss = self.mce_loss(logit, label)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, predict = logit.max(1)
                for qid, l in zip(ques_id, predict.cpu().numpy()):
                    quesid2ans[qid] = l

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def updated_dataset_for_self_train(self):
        # Update labels in dset for self_training
        # Create a new tuple of remaining data samples
        splits = 'self_train'
        leftover_tuple = get_tuple(splits, bs=args.batch_size, shuffle=False, drop_last=False)

        quesid2ans = self.predict(leftover_tuple)
        dset, loader, evaluator = leftover_tuple

        for qid, ans in quesid2ans.items():
                for datum in dset.data:
                    if datum['uid'] == qid:
                        datum['label'] = ans

        dset.splits = 'combined_train'
        return dset

    def reinitialize(self):
            # Update train_tuple
            dset = self.updated_dataset_for_self_train()
            tset = NLVR2TorchDataset(dset)
            evaluator = NLVR2Evaluator(dset)
            data_loader = DataLoader(
                tset, batch_size=args.batch_size,
                shuffle=True, num_workers=args.num_workers,
                drop_last=True, pin_memory=True
            )
            self.train_tuple = DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)

            del self.model
            self.setup_model()
            self.setup_optimizers()

if __name__ == "__main__":

    if args.mixmatch:
        nlvr2 = NLVR2MixMatch()
    else:
        nlvr2 = NLVR2()

    # Load Model
    if args.load is not None:
        nlvr2.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'hidden' in args.test:
            nlvr2.predict(
                get_tuple(args.test, bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'hidden_predict.csv')
            )
        elif 'test' in args.test or 'valid' in args.test:
            result = nlvr2.evaluate(
                get_tuple(args.test, bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, '%s_predict.csv' % args.test)
            )
            print(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        if args.train == 'train':
            print('Splits in Train data:', nlvr2.train_tuple.dataset.splits)

            if nlvr2.valid_tuple is not None:
                print('Splits in Valid data:', nlvr2.valid_tuple.dataset.splits)

            else:
                print("DO NOT USE VALIDATION")
            nlvr2.train(nlvr2.train_tuple, nlvr2.valid_tuple)

            # Self training
            if args.self_train is True and args.mixmatch is False:

                # Update the settings of the instance before retraining
                nlvr2.reinitialize()

                # Retrain with updated settings
                nlvr2.train(nlvr2.train_tuple, nlvr2.valid_tuple)
