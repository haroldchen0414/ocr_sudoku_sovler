# -*- coding: utf-8 -*-
# author: haroldchen0414

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
from sudoku import Sudoku
import numpy as np
import imutils
import cv2

class SudokuSolver:
    def __init__(self):
        pass

    def find_puzzle(self, image, debug=False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)
        thresh = cv2.bitwise_not(thresh)

        if debug:
            cv2.imshow("Thresh", thresh)
            cv2.waitKey(0)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        puzzleCnt = None

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                puzzleCnt = approx
                break

        if puzzleCnt is None:
            raise Exception("找不到数独框")
        
        if debug:
            output = image.copy()
            cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
            cv2.imshow("Puzzle Contour", output)
            cv2.waitKey(0)

        puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
        warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))

        if debug:
            cv2.imshow("Puzzle", puzzle)
            cv2.imshow("Warped", warped)
            cv2.waitKey(0)
        
        return (puzzle, warped)
    
    def extract_digit(self, cell, debug=False):
        thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = clear_border(thresh)

        if debug:
            cv2.imshow("Cell Thresh", thresh)
            cv2.waitKey(0)
        
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if len(cnts) == 0:
            return None
        
        c = max(cnts, key=cv2.contourArea)
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        (h, w) = thresh.shape
        percentFilled = cv2.countNonZero(mask) / float(w * h)

        if percentFilled < 0.03:
            return None
        
        digit = cv2.bitwise_and(thresh, thresh, mask=mask)

        if debug:
            cv2.imshow("Digit", digit)
            cv2.waitKey(0)
        
        return digit
    
    def fix_puzzle(self, puzzle, x, y, digit):
        # x, y为行列, 从0开始
        puzzle[x, y] = digit
        return puzzle
    
    def solve(self, image_path, debug=False):
        model = load_model("sudokuNet.h5")
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        image = cv2.resize(image, (600, 600))
        (puzzleImage, warped) = self.find_puzzle(image)
        board = np.zeros((9, 9), dtype="int")

        stepX = warped.shape[1] // 9
        stepY = warped.shape[0] // 9

        cellLocs = []

        for y in range(0, 9):
            row = []

            for x in range(0, 9):
                startX = x * stepX
                startY = y * stepY
                endX = (x + 1) * stepX
                endY = (y + 1) * stepY

                row.append((startX, startY, endX, endY))
                cell = warped[startY:endY, startX:endX]
                digit = self.extract_digit(cell)

                if digit is not None:
                    roi = cv2.resize(digit, (28, 28)).astype("float32") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    pred = model.predict(roi).argmax(axis=1)[0]
                    board[y, x] = pred

                    if debug:
                        cv2.imshow("Digit", digit)
                        print("Predicted digit: {}".format(str(pred)))
                        cv2.waitKey(0)

            cellLocs.append(row)

        # 手动修复数独, 行列从0开始, 例如test.jpg
        #self.fix_puzzle(board, 1, 8, 1)
        #self.fix_puzzle(board, 4, 7, 6)
        #self.fix_puzzle(board, 7, 8, 6)
        puzzle = Sudoku(3, 3, board=board.tolist())
        puzzle.show()

        solution = puzzle.solve()
        solution.show_full()

        for (cellRow, boardRow) in zip(cellLocs, solution.board):
            for (box, digit) in zip(cellRow, boardRow):
                startX, startY, endX, endY = box

                textX = int((endX - startX) * 0.33)
                textY = int((endY - startY) * -0.2)
                textX += startX
                textY += endY
                cv2.putText(puzzleImage, str(digit), (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Orignal", image)
        cv2.imshow("Result", puzzleImage)
        cv2.waitKey(0)

solver = SudokuSolver()
solver.solve('test1.jpg')
