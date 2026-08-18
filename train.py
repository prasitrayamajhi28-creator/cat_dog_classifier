import torch
import torch.nn as nn
from dataset import(
    train_loader,
    val_loader
)
from model import CatDogCNN

#device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#Create model
model = CatDogCNN(num_classes=2)

#freeze pretrain layers
for param in model.model.parameters():
    param.requires_grad=False
#unfreez layer4
for param in model.model.layer4.parameters():
    param.requires_grad = True
#Unfreeze final classifier
for param in model.model.fc.parameters():
    param.requires_grad=True

#Mocie model to device
model=model.to(device)

loss_fn=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([

    {
        "params": model.model.layer4.parameters(),
        "lr": 1e-5
    },

    {
        "params": model.model.fc.parameters(),
        "lr": 1e-4
    }

])

epochs=20
patience=5
epochs_without_improvement = 0
best_val_loss=float("inf")
train_losses=[]
val_losses=[]
#training loop
for epoch in range(epochs):
    model.train()
    total_train_loss=0
    correct_train=0
    total_train=0
    for X_batch,y_batch in train_loader:
        X_batch=X_batch.to(device)
        y_batch=y_batch.to(device)
        logists=model(X_batch)
        loss=loss_fn(logists,y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        #training accuracy
        predictions=torch.argmax(
            logists,dim=1
        )
        correct_train+=(
            predictions==y_batch
        ).sum().item()
        total_train += y_batch.size(0)
    avg_train_loss=(
        total_train_loss/
        len(train_loader)
    )
    train_accuracy=(correct_train/total_train)

    #validation
    model.eval()
    total_val_loss=0
    correct_val=0
    total_val=0
    with torch.no_grad():
        for X_batch,y_batch in val_loader:
            X_batch=X_batch.to(device)
            y_batch=y_batch.to(device)
            logists=model(X_batch)
            loss=loss_fn(logists,y_batch)
            total_val_loss += loss.item()
            predictions=torch.argmax(logists,dim=1)
            correct_val+=(predictions==y_batch).sum().item()
            total_val+=y_batch.size(0)
        avg_val_loss=(
                total_val_loss/
                len(val_loader)
            )
        val_accuracy=(correct_val/total_val)    
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    #save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(
            model.state_dict(),
            "best_cat_dog_model.pth"
        )
        print(
            f"Epoch {epoch + 1}: "
            f"Best model saved"
        )
    else:
        epochs_without_improvement += 1
        print(
            f"No improvement for "
            f"{epochs_without_improvement} "
            f"epoch(s)"
        )


    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: "
        f"{avg_train_loss:.4f} "
        f"Train Acc: "
        f"{train_accuracy * 100:.2f}% "
        f"Val Loss: "
        f"{avg_val_loss:.4f} "
        f"Val Acc: "
        f"{val_accuracy * 100:.2f}%"
    )

    if epochs_without_improvement >= patience:
        print("\nEarly stopping triggered.")
        print(
            f"Validation loss did not improve "
            f"for {patience} epochs."
        )
        break

print("\nTraining complete.")
print("Best validation loss:",best_val_loss)