## Steps for Repository Setup on Gypsum

```
git clone https://github.com/martiansideofthemoon/mixmatch_lxmert
cd mixmatch_lxmert
ln -s /mnt/nfs/work1/miyyer/kalpesh/projects/mixmatch/lxmert/data data
ln -s /mnt/nfs/work1/miyyer/kalpesh/projects/mixmatch/lxmert/snap snap
```

## Steps for Scheduling Jobs

Simply run `schedule_vqa.py` or `schedule_nlvr2.py`. It will adjust the usernames.

In case you have environment updates, change `vqa_finetune_template_vsuman.sh` or `nlvr2_finetune_template_vsuman.sh`.