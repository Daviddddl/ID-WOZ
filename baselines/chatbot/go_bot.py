from deeppavlov import build_model, configs

bot1 = build_model(configs.go_bot.gobot_dstc2, download=True)

bot1(['hi, i want restaurant in the cheap pricerange'])
bot1(['bye'])

bot2 = build_model(configs.go_bot.gobot_dstc2_best, download=True)

bot2(['hi, i want chinese restaurant'])
bot2(['bye'])