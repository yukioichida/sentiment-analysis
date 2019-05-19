import random
import torch
import torch.nn as nn
from torchtext import data
from torchtext import datasets
from model import RecurrentNetwork
from torch.optim import Adam

from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import ProgressBar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fixed random SEED
SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Definition of how we deal with text, given the features and associated label
TEXT = data.Field(include_lengths=True)
LABEL = data.LabelField(dtype=torch.float)

# In this case, we use IMDb dataset
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

# separate validation set for cross validation. using default split ratio which is 0.7
train_data, val_data = train_data.split(random_state=random.seed(SEED))

MAX_VOCAB_SIZE = 20_000

TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)
LABEL.build_vocab(train_data)

train_iterator, val_iterator, test_iterator = data.BucketIterator.splits((train_data, val_data, test_data),
                                                                         batch_size=32,
                                                                         device=device,
                                                                         sort_within_batch=True)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
RNN_N_LAYERS = 1
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]  # index of pad token
N_EPOCH = 10

model = RecurrentNetwork(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, RNN_N_LAYERS, DROPOUT, PAD_IDX)

n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {n_parameters} trainable parameters')

optimizer = Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)


def process_function(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, text_lengths = batch.text
    y = batch.label
    y_pred = model(x, text_lengths).squeeze(1)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def eval_function(engine, batch):
    model.eval()
    with torch.no_grad():
        x, text_lengths = batch.text
        y = batch.label
        y_pred = model(x, text_lengths).squeeze(1)
        return y_pred, y


trainer = Engine(process_function)
train_evaluator = Engine(eval_function)
validation_evaluator = Engine(eval_function)


def thresholded_output_transform(output):
    y_pred, y = output
    y_pred = torch.round(y_pred)
    return y_pred, y


RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
# validation
Accuracy(output_transform=thresholded_output_transform).attach(train_evaluator, 'accuracy')
Loss(criterion).attach(train_evaluator, 'bce')
# test
Accuracy(output_transform=thresholded_output_transform).attach(validation_evaluator, 'accuracy')
Loss(criterion).attach(validation_evaluator, 'bce')

pbar = ProgressBar(persist=True, bar_format="")
pbar.attach(trainer, ['loss'])

trainer.run(train_iterator, max_epochs=N_EPOCH)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc

def evaluate(iterator):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text

            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


test_loss, test_acc = evaluate(test_iterator)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
