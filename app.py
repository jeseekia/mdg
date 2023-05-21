!pip install -Uqq fastai

from fastai.vision.all import *
import gradio as gr


__all__ = ['is_cat', 'learn', 'classify_image', 'categories', 'image', 'label', 'examples', 'iface']
def greet(name):
    return "Hello " + name + "!!"

def is_cat(x): return x[0].isupper()


learn = load_learner('model.pkl')

categories = ('Dog', 'Cat')

def classify_image(img):
    pred,i,probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))


image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = ['cat-pic.jpeg']


iface = gr.Interface(fn=classify_image, inputs="image", outputs="label", examples=examples)
iface.launch(inline=False)