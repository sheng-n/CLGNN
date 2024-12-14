import copy
import time
import torch
import numpy as np
import torch.nn as nn
from layer import CLGNN_GCN
from layer import CLGNN_GIN
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, average_precision_score, f1_score, auc


def train_model(data_o, data_a, train_loader, test_loader, args):
    # extract features

    print('Loading model...')
    if args.model_type == 'CLGNN_GCN':
        model = CLGNN_GCN(feature=args.dimensions, hidden1=args.hidden1, hidden2=args.hidden2, decoder1=args.decoder1,
                          dropout=args.dropout)

    elif args.model_type == 'CLGNN_GIN':
        model = CLGNN_GIN(feature=args.dimensions, hidden1=args.hidden1, hidden2=args.hidden2, decoder1=args.decoder1,
                          dropout=args.dropout)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.BCELoss()
    b_xent = nn.BCEWithLogitsLoss()
    loss_history = []

    if args.cuda:
        model.to('cuda')
        data_o.to('cuda')
        data_a.to('cuda')

    # Train model
    lbl = data_a.y
    t_total = time.time()
    print('Start Training...')

    for epoch in range(args.epochs):
        t = time.time()
        print('-------- Epoch ' + str(epoch + 1) + ' --------')
        y_pred_train = []
        y_label_train = []

        for i, (label, inp) in enumerate(train_loader):
            if args.cuda:
                label = label.cuda()

            model.train()
            optimizer.zero_grad()
            output, cla_os, _, log1 = model(data_o, data_a, inp)

            log = torch.squeeze(m(output),1)
            loss1 = loss_fct(log, label.float())
            loss2 = b_xent(cla_os, lbl.float())

            loss_train = args.loss_ratio1 * loss1 + args.loss_ratio2 * loss2

            loss_history.append(loss_train.item())
            loss_train.backward()
            optimizer.step()

            label_ids = label.to('cpu').numpy()
            y_label_train = y_label_train + label_ids.flatten().tolist()
            y_pred_train = y_pred_train + log.flatten().tolist()

            if i % 100 == 0:
                print('epoch: ' + str(epoch + 1) + '/ iteration: ' + str(i + 1) + '/ loss_train: ' + str(
                    loss_train.cpu().detach().numpy()))

        roc_train = roc_auc_score(y_label_train, y_pred_train)

        print('epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'auroc_train: {:.4f}'.format(roc_train),
                  'time: {:.4f}s'.format(time.time() - t))

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    auroc_test, prc_test, f1_test, loss_test = test(model, test_loader, data_o, data_a, args)
    print('loss_test: {:.4f}'.format(loss_test.item()), 'auroc_test: {:.4f}'.format(auroc_test),
          'auprc_test: {:.4f}'.format(prc_test), 'f1_test: {:.4f}'.format(f1_test))


def test(model, loader, data_o, data_a, args):

    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.BCELoss()
    b_xent = nn.BCEWithLogitsLoss()

    model.eval()
    y_pred = []
    y_label = []
    lbl = data_a.y

    with torch.no_grad():
        for i, (label, inp) in enumerate(loader):

            if args.cuda:
                label = label.cuda()

            output, cla_os, _, log1 = model(data_o, data_a, inp)
            log = torch.squeeze(m(output), 1)

            loss1 = loss_fct(log, label.float())
            loss2 = b_xent(cla_os, lbl.float())
            loss = args.loss_ratio1 * loss1 + args.loss_ratio2 * loss2

            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + log.flatten().tolist()
            outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), loss
