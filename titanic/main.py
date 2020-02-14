import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
from titanic_net import Net
from utils import Utils, TitanicDataset
from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import csv
import pickle

def train(train_loader, model, optimizer, criterion, epochs):

    mdoel = model.double()
    for epoch in range(epochs):
        acc = 0
        for (X, y) in train_loader:

            optimizer.zero_grad()
            output = model(X)
            pred = [1 if out > 0.5 else 0 for out in output]
            pred = torch.tensor(pred)
            pred = pred.view(pred.shape[0], 1)
            for yp, yt in zip(pred, y):
                if yp == yt:
                    acc += 1

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        acc /= len(train_loader.dataset)

        print('Epoch {}    Loss: {}    Accuracy {}'.format(epoch+1, loss, acc))

def evaluate(loader, model):
    data = loader.dataset.data
    labels = loader.dataset.labels

    data = torch.tensor(data)
    labels = torch.tensor(labels)
    y_pred = model(data)

    acc = Utils().calculate_acc(y_pred, labels)
    print(f'Evaluation Accuracy: {acc}')

def predict(model, test_data):
    result = []
    result.append(['PassengerId', 'Survived'])
    y_pred = model.predict(test_data)
    for sample, yp in zip(test_data, y_pred):
        result.append([int(sample[0]), int(yp)])
    with open('submission.csv', 'w') as f:
        wr = csv.writer(f)
        wr.writerows(result)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', help='File containing data')
    parser.add_argument('--model', '-m', help='Select model for saving or evaluating')
    parser.add_argument('--eval', action='store_true', help='Evaluate model on data')
    parser.add_argument('--pred', action='store_true', help='Predict on data')
    args = parser.parse_args()

    if not args.file:
        filepath = 'data/train.csv'
    else:
        filepath = args.file

    model_type = args.model
    if not model_type:
        model_type = 'fcn'



    elif model_type == 'rf':
        model = RandomForestClassifier(max_depth=6, n_estimators=100)

    elif model_type == 'xgb':
        model = XGBClassifier()

    elif model_type == 'gbc':
        model = GradientBoostingClassifier()

    elif model_type == 'ada':
        base = RandomForestClassifier(max_depth=6, n_estimators=100)
        model = AdaBoostClassifier(base_estimator=base)

    elif model_type == 'svm':
        model = LinearSVC(max_iter=3000)

    elif model_type == 'bag':
        base = None
        model = BaggingClassifier(base_estimator=base)

    elif model_type == 'stack':
        clf1 = XGBClassifier(eta=0.1)
        clf2 = RandomForestClassifier()
        clf3 = GradientBoostingClassifier()
        clf4 = AdaBoostClassifier()

        estimators = [('xgb', clf1), ('rf', clf2), ('gbc', clf3), ('ada', clf4)]
        model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

    elif model_type == 'ensemble':
        clf1 = XGBClassifier(eta=0.1)
        clf2 = RandomForestClassifier()
        clf3 = GradientBoostingClassifier()
        clf4 = AdaBoostClassifier()

        estimators = [('xgb', clf1), ('rf', clf2), ('gbc', clf3), ('ada', clf4)]
        model = VotingClassifier(estimators=estimators, voting='hard')


    if args.eval:
        # evaluate()
        pass
    elif args.pred:
        test_data = Utils().process_data(filepath)
        model = pickle.load(open(f'models/{model_type}.sav', 'rb'))
        predict(model, test_data)

    else:
        # train model on data
        lr = 0.001
        epochs = 500
        decay = 1e-4
        batch_size = 32

        # load data and clean it up
        utils = Utils()
        data = utils.process_data(filepath)
        train_data, train_ys, val_data, val_ys = utils.split_data(data)

        # load data into custom dataset for dataloaders
        train_dataset = TitanicDataset(train_data, train_ys, to_normalize=True)
        val_dataset = TitanicDataset(val_data, val_ys, to_normalize=False)

        # load data into dataloaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

        # set up model
        if model_type == 'fcn':
            model = Net(input_channel=train_data.shape[1])
            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
            criterion = nn.BCELoss()
            #train
            train(train_loader, model, opt, criterion, epochs)
            #save model
            torch.save(f'models/{model_type}.pth')
            #evaluate
            evaluate(val_loader, model)

        else:
            train_ys = train_ys.reshape(train_ys.shape[0])
            val_ys = val_ys.reshape(val_ys.shape[0])

            model.fit(train_data, train_ys)
            pickle.dump(model, open(f'models/{model_type}.sav', 'wb'))
            y_pred = model.predict(val_data)
            acc = utils.calculate_acc(y_pred, val_ys)
            print(acc)



if __name__ == '__main__':
    main()
