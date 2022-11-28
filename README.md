# multiwoz-bahasa

## Pretraining BERT for Bahasa
```bash
cd multiwoz-bahasa/bert

bash create_pretrain_data.sh

bash run_pretraining.sh
```

## Statistics and prepare for dataset
```python
python datasets.py
```
## Datasets

| 数据 | 规模 | 链接 | 密码 |
| --- | --- | --- | --- |
| Bahasa BERT source corpus | 33M | https://pan.baidu.com/s/1aWdcjqfqUzfcgTUWkN5U_A <br> https://drive.google.com/open?id=1a-KmOQhYv0VguJfBVjhVbUgf_XH_RkX3| fvda |
| Bahasa Atome dictation | 10k | annotation-images/CallBot/20191020/mp3_uniq_uniq/ <br> annotation-images/CallBot/20190727_1/ <br> annotation-images/CallBot/20170807/ | 张新田 |
| BahasaWOZ | 10k | https://pan.baidu.com/s/1Wbh0B8o63T2S52_FkdaxjQ <br> https://drive.google.com/open?id=1bbJapCwdN3TsTnTgx88hqIUyXinDNeDH | gb5x |
| Free templates dialogue | 3k | https://pan.baidu.com/s/1ac-GJfrAqqtxhYdEK3BI5A | 6kvo |
| En-2-ID | 1k | https://pan.baidu.com/s/1d39YczRS-MqZ8e7iF8M-ug | avsb |
| ID-2_En | 1k | https://pan.baidu.com/s/1d39YczRS-MqZ8e7iF8M-ug | avsb |
