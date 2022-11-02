import tkinter as tk

from inference import one_shot_inference

class FrameApp(tk.Frame):
    def __init__(self, master):
        super(FrameApp, self).__init__(master)

        self.model_dir = "./attnS2S_atis_model"
        self.inputSentence = []
        self.predict_intent = []
        self.slot_list = []

        self.grid()
        self.run_flag = True

        self.textLabel = tk.Label(self, text="Input Sentence")
        self.textLabel.grid(row=2, column=1)
        self.textButton2 = tk.Button(self, height=1, width=10, text="Read", command=self.getTextInput)
        self.textButton2.grid(row=2, column=2)
        self.textInputBlock = tk.Text(self, height=4)
        self.textInputBlock.grid(row=3, column=1)

        self.textOutputBlock1 = tk.Text(self, height=4)
        self.textOutputBlock1.grid(row=4, column=1)
        self.textLabel = tk.Label(self, text="Predicted Intent")
        self.textLabel.grid(row=5, column=1)
        self.textButton3 = tk.Button(self, height=1, width=10, text="Inference", command=self.showTextOutput)
        self.textButton3.grid(row=5, column=1)

        self.textOutputBlock2 = tk.Text(self, height=4)
        self.textOutputBlock2.grid(row=6, column=1)
        self.textLabel = tk.Label(self, text="Predicted Slot")
        self.textLabel.grid(row=7, column=1)
        self.textOutputBlock3 = tk.Text(self, height=10)
        self.textOutputBlock3.grid(row=8, column=1)

    def getModelPath(self):
        self.model_dir = "./attnS2S_atis_model"

    def clearText(self):
        self.textOutputBlock1.delete("1.0", "end")
        self.textOutputBlock2.delete("1.0", "end")
        self.textOutputBlock3.delete("1.0", "end")

    def getTextInput(self):
        self.clearText()
        self.inputSentence = self.textInputBlock.get(1.0, tk.END+"-1c")

        # transfer input sentence into words in lines
        lines = []
        line = self.inputSentence.replace(' ', '')
        words = [char for char in line]
        lines.append(words)

        self.inputSentence = lines
        self.textOutputBlock1.insert(tk.END, self.inputSentence)

    def showTextOutput(self):

        self.predict_intent, self.slot_list = one_shot_inference(self.inputSentence, self.model_dir)

        self.textOutputBlock2.insert(tk.END, self.predict_intent)
        self.textOutputBlock3.insert(tk.END, self.slot_list)

    def _quit(self):
        super(FrameApp, self).quit()
        self.run_flag = False

root = tk.Tk()
root.geometry("680x550")
app = FrameApp(root)

while app.run_flag:
    # runs mainloop of program
    app.update()

app.destroy()
