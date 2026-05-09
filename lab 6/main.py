import numpy as np
import matplotlib.pyplot as plt


# 1. Навчальна пара
inputs = np.array([[0, 1, 1],
                   [0, 0, 0],
                   [1, 0, 0],
                   [1, 1, 1],
                   [0, 0, 1]])

outputs = np.array([[0],
                    [0],
                    [1],
                    [1],
                    [1]])

# 2. Конструювання нейрономережі
class NeuralNetwork:
    # ініціалізувати змінні в класі
    def __init__(self, inputs, outputs):
        self.inputs  = inputs
        self.outputs = outputs
        # Вагові коефіціенти
        self.weights = np.array([[0.50], [0.50], [0.50]])
        self.error_history = []
        self.epoch_list = []

    # функція активації показова f(x) = 1 / (1 + e^(-x))
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    # подання вхідних даних до мережі
    def feed_forward(self):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))

    # повернення даних після мережі - похибка
    def backpropagation(self):
        self.error  = self.outputs - self.hidden
        delta = self.error * self.sigmoid(self.hidden, deriv=True)
        self.weights += np.dot(self.inputs.T, delta)

    # тренування мережі на 25 ітераціях (ЕПОХАХ)
    def train(self, epochs=25000):
        for epoch in range(epochs):
            self.feed_forward()
            self.backpropagation()
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

    # прогнозовані дані
    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction

# ініціалізація мережі
NN = NeuralNetwork(inputs, outputs)

# навчання мережі
NN.train()

# 3. Перевірка на навчальній вибірці
print('Перевірка на навчальній вибірці:')
for i in range(len(inputs)):
    pred = NN.predict(inputs[i].reshape(1, -1))[0][0]
    target = outputs[i][0]
    ok = '+' if round(pred) == target else '-'
    print(f'  вхід {inputs[i]}  ->  {pred:.4f} | ({target})  {ok}')

# 4. нові вхідні дані
print()
print('Доведення працездатності - нові вхідні дані:')
new_inputs = [
    np.array([[1, 0, 0]]),
    np.array([[0, 1, 1]]),
    np.array([[1, 1, 0]]),
]
for ni in new_inputs:
    pred = NN.predict(ni)[0][0]
    klass = round(pred)
    print(f'  вхід {ni[0]}  ->  {pred:.4f} | ({klass})')

print()
print(f'Похибка після навчання: {NN.error_history[-1]:.6f}')
print(f"Навчених епох: {len(NN.epoch_list)}")

# динаміка зміни помилки з часом тренування
plt.figure(figsize=(8, 4))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()