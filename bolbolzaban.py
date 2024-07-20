from transformers import pipeline, AutoTokenizer, GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained('bolbolzaban/gpt2-persian')
model     = GPT2LMHeadModel.from_pretrained('bolbolzaban/gpt2-persian')
generator = pipeline('text-generation', model, tokenizer=tokenizer, config={'max_length':256})
#sample    = generator('در یک اتفاق شگفت انگیز، پژوهشگران')
#sample    = generator('به نام خداوند جان و خرد کزین ')
#sample    = generator('آش با ')
#sample    = generator('من نگویم که مرا از قفس ')
sample    = generator('برو دنبال نون باش که ')
print(sample)
