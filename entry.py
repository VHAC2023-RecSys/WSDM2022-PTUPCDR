import os
from preprocessing import DataPreprocessingMid, DataPreprocessingReady
from run import Run
from args import prepare

if __name__ == "__main__":
    config_path = "config.json"
    args, config = prepare(config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.process_data_mid:
        for dealing in ["Books", "CDs_and_Vinyl", "Movies_and_TV"]:
            DataPreprocessingMid(config["root"], dealing).main()
    if args.process_data_ready:
        for ratio in [[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]]:
            for task in ["1", "2", "3"]:
                DataPreprocessingReady(
                    config["root"], config["src_tgt_pairs"], task, ratio
                ).main()
    print(
        "task:{}; model:{}; ratio:{}; epoch:{}; lr:{}; gpu:{}; seed:{};".format(
            args.task,
            args.base_model,
            args.ratio,
            args.epoch,
            args.lr,
            args.gpu,
            args.seed,
        )
    )

    if not args.process_data_mid and not args.process_data_ready:
        Run(config).main()
