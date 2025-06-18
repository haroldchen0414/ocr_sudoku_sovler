# ocr_sudoku_sovler
数独求解器

首先运行train_nn.py训练一个能识别数字的模型, 然后运行sudoku_solver.py  
例如test1.jpg
![image](https://github.com/user-attachments/assets/a9dc9faf-704d-41a9-8d5b-2a5310e45fc2)

对于test.jpg
![image](https://github.com/user-attachments/assets/12af9cec-b742-43a5-80a9-148ece6e0661)
第2行第9个数字1被识别成了7  
第5航第8个数字被识别成了0  
第5行第9个数字被是被成了0  
对于这种情况，可以使用fix_puzzle(self, puzzle, x, y, digit)来修复, 其中x, y代表行和列, 从0开始    
self.fix_puzzle(board, 1, 8, 1)  
self.fix_puzzle(board, 4, 7, 6)  
self.fix_puzzle(board, 7, 8, 6)  
修复之后, 可以得到结果  
![image](https://github.com/user-attachments/assets/b998a5f5-7043-4584-817e-f35cfe9752b9)





