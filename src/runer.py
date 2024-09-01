import config.model
from model.pipeline.vit import ViT
from model.pipeline.data_setup import download_data, create_dataloaders
from model.pipeline import engine
from torchvision import transforms
from torchinfo import summary
import torch
import config

from loguru import logger

LR = config.model_settings.lr
EPOCHS = config.model_settings.epochs
BATCH_SIZE = config.model_settings.batch_size
IMAGE_SIZE = config.model_settings.image_size

device = "cuda" if torch.cuda.is_available() else "cpu"

@logger.catch
def main():

    image_path = download_data(
        source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
        destination="pizza_steak_sushi",
    )

    # Setup directory paths to train and test images
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    manual_transforms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )

    logger.info(f"Manually created transforms: {manual_transforms}")

    train_dataloader, test_dataloader, class_names = create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=manual_transforms,
        batch_size=BATCH_SIZE,
    )
    logger.info(
        f"Data Preparation with transforms: {[train_dataloader, test_dataloader, class_names]}"
    )

    vit = ViT(num_classes=3)

    summary(
        model=vit,
        input_size=(32, 3, 224, 224),
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            "params_percent",
            "kernel_size",
            "mult_adds",
            "trainable",
        ],
        col_width=20,
        row_settings=["var_names"],
    )

    optimizer = torch.optim.Adam(
        params=vit.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.1
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    results = engine.train(
        model=vit,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=EPOCHS,
        device=device,
    )


if __name__ == "__main__":
    main()
