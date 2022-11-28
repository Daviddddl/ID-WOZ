from deeppavlov import build_model, configs

PIPELINE_CONFIG_PATH = configs.ner.slotfill_dstc2
slotfill_model = build_model(PIPELINE_CONFIG_PATH, download=True)
slotfill_model(['I would like some chinese food', 'The west part of the city would be nice'])
