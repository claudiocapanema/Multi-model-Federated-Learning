import fiftyone as fo
import torchvision
dataset =fo.zoo.load_zoo_dataset(
    name='imagenet-2012',
    source_dir="arquivo/",
    classes=["dog", "cat", "bird"]  # Selecionar as classes desejadas
)
new_dataset = dataset.select(fields=["tags"])  # Selecionar apenas o campo "tags"
new_dataset.save("imagenet_subset")  # Salvar o subconjunto com o nome "imagenet_subset"
new_dataset = fo.load_dataset("imagenet_subset")
print(new_dataset.info())  # Exibir informações sobre o dataset
new_dataset.view_sample()  # Visualizar uma amostra aleatória do dataset
