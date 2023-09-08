import streamlit as st
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def function(x):
    return x**2 - 10 * torch.cos(2 * np.pi * x) + 10
torch.manual_seed(42)
x = torch.randn(1, requires_grad=True)

def main():
    st.title("Function Optimization")

    optimizer_options = ["SGD", "Adam", "RMSprop"]
    selected_optimizer = st.sidebar.selectbox("Select Optimizer", optimizer_options)

    num_epochs = st.sidebar.number_input("Number of Epochs", value=1000)
    learning_rate = st.sidebar.number_input("Learning Rate", value=0.1)

    if selected_optimizer == "SGD":
        momentum = st.sidebar.number_input("Momentum", value=0.9)
        optimizer = optim.SGD([x], lr=learning_rate, momentum=momentum)
    elif selected_optimizer == "Adam":
        b1 = st.sidebar.number_input("b1 (beta1)", value=0.9)
        b2 = st.sidebar.number_input("b2 (beta2)", value=0.999)
        optimizer = optim.Adam([x], lr=learning_rate, betas=(b1, b2))
    elif selected_optimizer == "RMSprop":
        alpha = st.sidebar.number_input("Alpha", value=0.99)
        optimizer = optim.RMSprop([x], lr=learning_rate, alpha=alpha)

    st.write(f"Using optimizer: {selected_optimizer}")
    st.write(f"Number of Epochs: {num_epochs}")
    st.write(f"Learning Rate: {learning_rate}")

    losses = []
    x_values = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = function(x)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        x_values.append(x.item())

    final_loss = function(x).item()
    st.write(f"Final Loss: {final_loss:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Optimization")
    ax1.plot(losses)

    x_vals = torch.linspace(-10, 10, 100)
    y_vals = function(x_vals)
    ax2.plot(x_vals.numpy(), y_vals.numpy(), label="Function Curve")
    ax2.scatter(x_values, [function(torch.tensor([x])).item() for x in x_values], color="red", marker="x", label="Optimization Variable (x)")
    ax2.annotate(f'x = {x.item():.4f}\nf(x) = {function(x).item():.4f}',
                 xy=(x.item(), function(x).item()), xycoords='data',
                 xytext=(-50, 30), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    ax2.set_xlabel("x")
    ax2.set_ylabel("f(x)")
    ax2.legend()

    st.pyplot(fig)

if __name__ == "__main__":
    main()
