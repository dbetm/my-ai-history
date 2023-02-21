import gradio as gr
from fastai.vision.all import load_learner


CATS_MAP = {
    "picasso": "Pablo Picasso",
    "vanGogh": "Vincent van Gogh",
    "dali": "Salvador Dalí",
    "daVinci": "Leonardo da Vinci",
    "rembrandt": "Rembrandt",
}

CATS_MAP_V2 = {
    "picasso": "Pablo Picasso",
    "vanGogh": "Vincent van Gogh",
    "dali": "Salvador Dalí",
    "daVinci": "Leonard da Vinci",
    "rembrandt": "Rembrandt",
    "monet": "Claude Monet",
    "caruso": "Santiago Caruso",
    "renoir": "Pierre-Auguste Renoir",
    "oKeeffe": "Georgia O’Keeffe",
    "krasner": "Lee Krasner",
}

# load pre-trained model
model = load_learner("model_v2.pkl")

# get classes name in right order
full_name_cats = [CATS_MAP_V2[key_class] for key_class in model.dls.vocab]


def classify_image(img) -> dict:
    category, idx, probs = model.predict(img)

    return dict(zip(full_name_cats, map(float, probs)))


# Gradio control
image = gr.inputs.Image(shape=(224, 224))
label = gr.outputs.Label(num_top_classes=4)
examples = [
    f"images_examples/{filename}" 
    for filename in ("mona_lisa.jpg", "starry_night.jpg", "persistence_memory.jpg")
]

painters_list = CATS_MAP_V2.values()

gui = gr.Interface(
    fn=classify_image,
    inputs=image,
    outputs=label,
    examples=examples,
    title="Detect the painter",
    description=(
        f"Detect if the given painting image is by a famous painter ({painters_list})."
    )
)
gui.launch(inline=False)