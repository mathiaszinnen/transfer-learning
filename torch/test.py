
from data.odor_dataset import get_dataset
from transforms import get_test_transforms
from utils import load_model, prepare_dataloader, write_preds
from trainer import get_test_trainer


def main(args):
    test_ts = get_test_transforms(size=400)
    test_ds = get_dataset(args.test_imgs, args.test_anns, test_ts)
    test_data = prepare_dataloader(test_ds, 1, False)
    model, optimizer = load_model(test_ds.num_classes, 0.0001)
    trainer = get_test_trainer(model, test_data, args.load_model_path)
    outputs = trainer.predict()
    if args.preds_pth is not None:
        write_preds(args.preds_pth, test_ds.coco, outputs)


    print('a')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model_path', help='Path to model')
    parser.add_argument('--test_imgs', help='Path to test images')
    parser.add_argument('--test_anns', help='Path to test annotations')
    parser.add_argument('--preds_pth', help='Path to write the generated predictions. '
                                            'Omit to do just the evaluation.', required=False)
    args = parser.parse_args()

    main(args)