import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from Trainer import *
from utils import *
from models import *
from args import *

if __name__ == "__main__":

    args = make_args()

    print(args)

    # verbose = True #
    verbose = False #
    

    device = torch.device(f"cuda:{args.cuda}")

    model_path = get_saved_model_path(args)

    print("Model will be saved at ", model_path)

    X, edge_index = load_data(args)
    X = coo2torch(X).to_dense()
    edge_index = torch.tensor(edge_index)
    nodes_num, attr_num = X.shape

    print(
        f"{args.dataset}: nodes_num:{nodes_num}, attr_num:{attr_num}, edges_num:{edge_index.shape[1]}"
    )

    topk = args.topk

    kw = f"kw{args.kw}"

    gt = load_gt(f"./data/{args.dataset}/test_gt/{kw}/{kw}_top{str(topk)}_output.txt")

    val_gt = gt

    # val_gt = load_gt(f"./data/{args.dataset}/val_gt/{kw}/{kw}_top{str(topk)}_output.txt")

    qs = []
    ans = []
    for q in gt.keys():
        qs.append(str2int(q))
        ans.append(gt[q])

    val_qs = []
    val_ans = []
    for q in val_gt.keys():
        val_qs.append(str2int(q))
        val_ans.append(val_gt[q])

    # -------------#

    qX = queries2tensor(qs, attr_num).to(device)
    val_qX = queries2tensor(val_qs, attr_num).to(device)

    X = X.to(device)
    edge_index = edge_index.to(device)

    output_dict = {}

    for repeat in range(args.repeat):

        model = KSNN(
            X.shape[1],
            args.hid_d,
            args.d,
            layer_num=args.layer_num,
            conv_num=args.conv_num,
            alpha=args.alpha,
        ).to(device)

        trainer = Trainer(model, X, edge_index, args)

        for e in range(args.epochs_num):
            # print("Epoch: {}".format(e))

            trainer.train_batch()

            if e % 5 == 0:
                trainer.decay_learning_rate(e, args.lr)

            if e % args.eval_time == 0:
                # train set

                hit_valid = trainer.test(val_qX, val_ans, topk)

                if hit_valid <= trainer.bestResult:
                    time += 1
                else:
                    trainer.bestResult = hit_valid
                    time = 0
                    trainer.save(model_path)
                if time > 20:
                    print(("BEST RESULT ON VALIDATE DATA:{:.4}").format(trainer.bestResult))
                    break

        

        if os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path))

        trainer.model = model

        output_dict["model"] = output_dict.get('model',[]) + [trainer.test(qX, ans, topk, verbose=verbose)]

        print(f'repeat {repeat}: {output_dict["model"][-1]}')

    
    output_dict["model"] = np.array(output_dict["model"]).mean()

    print(f"Test Result:{output_dict['model']:.4}")

    if args.pca_verbal:

        model = KSNN(
            X.shape[1],
            args.hid_d,
            args.d,
            layer_num=args.layer_num,
            conv_num=args.conv_num,
            alpha=args.alpha,
        ).to(device)

        trainer = Trainer(model, X, edge_index, args)

        for e in range(args.epochs_num):

            trainer.train_sage()

            if e % 10 == 0:
                trainer.decay_learning_rate(e, args.lr)

            if e % args.eval_time == 0:
                hit_valid = trainer.sage_test(val_qX, val_ans, topk)
                if hit_valid <= trainer.bestResult:
                    time += 1
                else:
                    trainer.bestResult = hit_valid
                    time = 0
                    trainer.save(model_path)
                if time > 20:
                    # print(("Sage RESULT ON VALIDATE DATA:{:.4}").format(trainer.bestResult))
                    break

        if os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path))

        trainer.model = model
        output_dict["sage"] = trainer.sage_test(qX, ans, topk, verbose=verbose)
        print(f"Sage Result:{output_dict['sage']:.4}")

        output_dict["pca_conv"] = eval_Z(
            trainer.model.pca(X, edge_index),
            qX @ trainer.model.pca_v,
            ans,
            k=topk,
            verbose=verbose,
        )
        # Conv_PCA convolutional results
        print(f"PCA_conv:{output_dict['pca_conv']:.4}")

        output_dict["rpca_conv"] = eval_Z(
            trainer.model.rpca(X, edge_index),
            qX @ trainer.model.pca_v,
            ans,
            k=topk,
            verbose=verbose,
        )
        # Conv_rPCA convolutional results
        print(f"rPCA_conv:{output_dict['rpca_conv']:.4}")

        # directly use PCA
        output_dict["pca"] = eval_PCA(X, qX, ans, k=topk, verbose=verbose)

        print(f"Direct PCA:{output_dict['pca']:.4}")

    print(output_dict)

