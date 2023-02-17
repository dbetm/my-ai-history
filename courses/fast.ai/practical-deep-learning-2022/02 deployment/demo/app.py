import gradio as gr
from fastai.vision.all import load_learner


CATS_MAP = {
    "picasso": "Pablo Picasso",
    "vanGogh": "Vincent van Gogh",
    "dali": "Salvador Dalí",
    "daVinci": "Leonardo da Vinci",
    "rembrandt": "Rembrandt",
}

# load pre-trained model
model = load_learner("model.pkl")

# get classes name in right order
full_name_cats = [CATS_MAP[key_class] for key_class in model.dls.vocab]


def classify_image(img) -> dict:
    category, idx, probs = model.predict(img)
    return dict(zip(full_name_cats, map(float, probs)))


# Gradio control
image = gr.inputs.Image(shape=(224, 224))
label = gr.outputs.Label()
examples = [
    f"images_examples/{filename}" 
    for filename in ("mona_lisa.jpg", "starry_night.jpg", "le_reve.jpg")
]

gui = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
gui.launch(inline=False)