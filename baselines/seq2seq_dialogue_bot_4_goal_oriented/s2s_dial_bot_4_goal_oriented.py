from deeppavlov import build_model, configs

bot = build_model(configs.seq2seq_go_bot.bot_kvret, download=True)

dialog_id = '2b77c100-0fec-426a-a483-04ac03763776'
bot(['Hi! Where is the nearest gas station?'], [dialog_id])
bot(['Thanks, bye'], [dialog_id])
