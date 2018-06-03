import numpy as np
import os
from tkinter import *
import tkinter.ttk as ttk
import time, threading

root = Tk()
root.geometry('300x200')
root.title('Progressbar')

pbar = StringVar()

label = Label(textvariable=pbar, padx="30", pady="15")

pb = ttk.Progressbar(root,length = 200, mode="determinate")
pb.pack()

#  сигмоидальная функция приведения чисел (sigmoid)
def nonlin(x, seed=False): # превращает целое число в вывод нейросети
    if seed:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# объявляем переменную Х как двумерный массив
x = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]]) # в Х запишется транспонированная матрица

# делаем новый массив, в котором будут выводы
y = np.array([[0, 1, 1, 1]]).T # T - операция транспонирования, чтобы избежать путаницы при выводе

# расчеты нейросети
# создаем зерно генерации, чтобы сделать функцию детерминированной
np.random.seed(1)
# задаем семантический вес (случайным образом)
syn0 = 2 * np.random.random((3, 1)) - 1 # нулевой синаптический вес
# узел = иттерации
iterations = int(input("Введите количество иттераций: "))

for iter in range (iterations):
    l0 = x # сделаем копию вводного массива х, чтобы функция была чистой, то есть с ней можно работать независимо от входного массива
    l1 = nonlin(np.dot(l0, syn0)) # перераспределенные значения входного массиваб к вводному числу приклеили семантический вес, это будет выводной слой
    l1_error = y - l1 # переменная, которая хранит коэффициент ошибки
    l1_delta = l1_error * nonlin(l1, True)
# обновляем семантический вес
    syn0 += np.dot(l0.T, l1_delta)


def progressb():
    for iter in range(iterations):
        pb['value'] = (iter/iterations) * 100
        pbar.set(int(pb['value']))
    #    time.sleep(.1)

threading.Thread(target=progressb).start()
label.pack()

label_result = Label(text = l1, background="#705", foreground="#ccc", font=("Verdana", 10, "bold"))
label_result.pack()
root.mainloop()
    #    pb.step(10)
    #    root.mainloop()
    #    print(str(int(progress)) + "%")
    #    print("=" * int(progress))

    # print(str((iter/iterations) * 100) + "%")
# вывод результата
print("Вывод после тренировки")
print (l1)
