'''
Created on 

@author: Raja CSP

source:

'''

# importing the required libraries:
import torch
from transformers import BertTokenizer, BertForSequenceClassification

#  define color naming dataset. Here's an example dataset that includes color names and their corresponding RGB values:
colors = [    {'name': 'red', 'rgb': (255, 0, 0)},    {'name': 'green', 'rgb': (0, 255, 0)},    {'name': 'blue', 'rgb': (0, 0, 255)},    {'name': 'yellow', 'rgb': (255, 255, 0)},    {'name': 'purple', 'rgb': (128, 0, 128)},    {'name': 'orange', 'rgb': (255, 165, 0)},    {'name': 'black', 'rgb': (0, 0, 0)},    {'name': 'white', 'rgb': (255, 255, 255)}]

color_labels = {c['name']: i for i, c in enumerate(colors)}


#  define tokenizer and load a pre-trained BERT model:

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(colors))



# Next, we'll prepare our dataset by encoding the color names as input sequences and their labels as targets:
inputs = [tokenizer.encode(c['name'], add_special_tokens=True) for c in colors]
targets = [color_labels[c['name']] for c in colors]

# We'll use PyTorch's DataLoader to create batches of input/target pairs for training:

dataset = torch.utils.data.TensorDataset(torch.LongTensor(inputs), torch.LongTensor(targets))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

# We can now fine-tune the BERT model on our color naming dataset:
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
for epoch in range(5):
    total_loss = 0
    for batch in dataloader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs, labels=targets)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1} loss: {total_loss}')

# Once the model is trained, we can use it to predict the colors in new text by passing the text through the model and extracting the predicted color labels or probabilities:
text = 'The walls are painted yellow, and the curtains are green. Then it turned fuchsia and sienna and sky blue. Finally I found out it is Lavender and Lilac'
encoded_text = tokenizer.encode(text, add_special_tokens=True)
outputs = model(torch.LongTensor([encoded_text]))

print(type(outputs.logits))

# predicted_label = torch.argmax(outputs.logits).item()
# predicted_color = colors[predicted_label]['name']
# print(predicted_color)

