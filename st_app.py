import streamlit as st
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the Rastrigin function
def function(x):
    return x**2 - 10 * torch.cos(2 * np.pi * x) + 10

x = torch.randn(1, requires_grad=True)

# Streamlit app
def main():
    st.title("Rastrigin Function Optimization")

    # Sidebar options
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

    # Create a single figure and axes for both plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    col1, col2 = st.columns(2)  # Separate the columns for visualization

    # Initialize plots
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Optimization")

    x_vals = torch.linspace(-10, 10, 100)
    y_vals = function(x_vals)
    ax2.plot(x_vals.numpy(), y_vals.numpy(), label="Function Curve")
    ax2.set_xlabel("x")
    ax2.set_ylabel("f(x)")
    ax2.legend()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = function(x)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Update the plots
        ax1.clear()
        ax1.plot(losses)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss Optimization")

        ax2.clear()
        ax2.plot(x_vals.numpy(), y_vals.numpy(), label="Function Curve")
        ax2.scatter(x.item(), function(x).item(), color="red", marker="x", label="Optimization Variable (x)")
        ax2.annotate(f'x = {x.item():.4f}\nf(x) = {function(x).item():.4f}',
                     xy=(x.item(), function(x).item()), xycoords='data',
                     xytext=(-50, 30), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        ax2.set_xlabel("x")
        ax2.set_ylabel("f(x)")
        ax2.legend()

        plt.pause(0.01)  # Pause to allow time for the plot to update

    final_loss = function(x).item()
    st.write(f"Final Loss: {final_loss:.4f}")

    # Display the final updated figure within each column
    with col1:
        st.pyplot(fig)

    with col2:
        st.pyplot(fig)

if __name__ == "__main__":
    losses = []  # Initialize losses list
    main()
