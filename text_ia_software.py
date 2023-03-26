import sys, threading, os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QTableWidgetItem,QFileDialog, QDialog,QCompleter, QTableWidget, QTextEdit,QPushButton, QHBoxLayout, QVBoxLayout, QLineEdit, QMainWindow, QComboBox, QMessageBox, QGridLayout, QStackedLayout, QFormLayout, QScrollArea
from PyQt5.QtGui import QMovie, QPalette, QColor, QIntValidator, QRegExpValidator
from PyQt5.QtCore import QRegExp
################################### First Loading  #####################################
app = QApplication([])
window = QWidget()
window.setWindowTitle("GUI-NLP")
app.setStyle('Fusion')
################################### import Network Libs ################################
from happytransformer import HappyGeneration
from happytransformer.happy_generation import GenerationResult
from happytransformer import GENSettings
from happytransformer import HappyQuestionAnswering
from happytransformer import HappyTextToText, TTSettings
from happytransformer.happy_text_to_text import TextToTextResult
################################### Variable globales ##################################

loader = QDialog()
loaderlab=QVBoxLayout()
loader.setWindowTitle("Running")
loadermovie = QMovie("loader.gif")
loadermovie.start()
loaderlabel=QLabel()
loaderlabel.setMovie(loadermovie)
loaderlab.addWidget(loaderlabel)
loader.setLayout(loaderlab)
################################### Functions ##########################################
#################### intercalaires #######################
def fct_int_gen():
    fond.setCurrentIndex(0)

def fct_int_conv():
    fond.setCurrentIndex(1)

def fct_int_quest():
    fond.setCurrentIndex(2)

def fct_int_tran():
    fond.setCurrentIndex(3)
################## Generation text #######################
class GenText:
    def __init__(self):
        self.resogen=HappyGeneration()
        self.genresult=GenerationResult(text='coucou')
        self.generation_gui()

    def generation_gui(self):
        lgeneration = QVBoxLayout()
        lgeneration.addWidget(QLabel("Network model choice"))
        lgenreso=QFormLayout()
        self.genresoname = QLineEdit()
        genresos=["rien"]
        ee=""
        with open('txtgen.conf','r') as f:
            ee+=f.read()
        f.close()
        genresos=ee.split("\n")
        self.genresoname.setCompleter(QCompleter(genresos,self.genresoname))
        lgenreso.addRow("Network Name",self.genresoname)
        genresocharge = QPushButton("Take it!")
        genresocharge.clicked.connect(self.charger_resogen)
        lgenreso.addRow("Download and charge network",genresocharge)
        genreso = QWidget()
        genreso.setLayout(lgenreso)
        lgeneration.addWidget(genreso)
        lgeneration.addWidget(QLabel("Generation parameters"))
        lgenparam = QFormLayout()
        self.genno_repeat_ngram_size = QLineEdit()
        self.genno_repeat_ngram_size.insert("2")
        self.genno_repeat_ngram_size.setValidator(QIntValidator())
        lgenparam.addRow("no_repeat_ngram_size",self.genno_repeat_ngram_size)
        self.gentop_k = QLineEdit()
        self.gentop_k.insert("50")
        self.gentop_k.setValidator(QIntValidator())
        lgenparam.addRow("top_k",self.gentop_k)
        self.gentemperature = QLineEdit()
        self.gentemperature.insert("0.9")
        self.gentemperature.setInputMask("9.99")
        lgenparam.addRow("temperature",self.gentemperature)
        self.genmax_length = QLineEdit()
        self.genmax_length.insert("200")
        self.genmax_length.setValidator(QIntValidator())
        lgenparam.addRow("max_length",self.genmax_length)
        genparam = QWidget()
        genparam.setLayout(lgenparam)
        lgeneration.addWidget(genparam)
        lgeneration.addWidget(QLabel("Input text"))
        self.geninput = QTextEdit()
        lgeneration.addWidget(self.geninput)
        bgen = QPushButton("Generate")
        bgen.clicked.connect(self.gen_go)
        lgeneration.addWidget(bgen)
        lgeneration.addWidget(QLabel("Output text"))
        self.genoutput = QTextEdit()
        self.genoutput.setReadOnly(True)
        lgeneration.addWidget(self.genoutput)
        self.fgeneration = QWidget()
        self.fgeneration.setLayout(lgeneration)

    def gui(self):
        return self.fgeneration

    def tcharger_resogen(self):
        bidule=self.genresoname.text().split(":")
        letyp=bidule[0]
        lename=bidule[1]
        self.resogen=HappyGeneration(letyp,lename)
        loader.close()

    def charger_resogen(self):
        taction = threading.Thread(target=self.tcharger_resogen)
        taction.start()
        loader.exec_()

    def tgen_go(self):
        args = GENSettings(no_repeat_ngram_size=int(self.genno_repeat_ngram_size.text()), do_sample=True, early_stopping=False, top_k=int(self.gentop_k.text()), temperature=float(self.gentemperature.text()),max_length=int(self.genmax_length.text()))
        self.genresult = self.resogen.generate_text(self.geninput.toPlainText(), args=args)
        print(self.genresult.text)
        loader.close()

    def gen_go(self):
        traction = threading.Thread(target=self.tgen_go)
        traction.start()
        loader.exec_()
        self.genoutput.clear()
        self.genoutput.append(self.genresult.text)
###################### Conversation ############################
class Conversation:
    def __init__(self):
        self.resogen=HappyGeneration()
        self.genresult=GenerationResult(text='coucou')
        self.generation_gui()

    def generation_gui(self):
        self.layout = QVBoxLayout()
        self.layout.addWidget(QLabel('Network Model Choice'))
        lgenreso=QFormLayout()
        self.genresoname = QLineEdit()
        genresos=["rien"]
        ee=""
        with open('conv.conf','r') as f:
            ee+=f.read()
        f.close()
        genresos = ee.split("\n")
        self.genresoname.setCompleter(QCompleter(genresos,self.genresoname))
        lgenreso.addRow("Network Name",self.genresoname)
        genresocharge = QPushButton("Take it!")
        genresocharge.clicked.connect(self.charger_resogen)
        lgenreso.addRow("Download and load",genresocharge)
        wgenreso = QWidget()
        wgenreso.setLayout(lgenreso)
        self.layout.addWidget(wgenreso)
        self.layout.addWidget(QLabel('Generation Parameters'))
        lgenparam = QFormLayout()
        self.genno_repeat_ngram_size = QLineEdit()
        self.genno_repeat_ngram_size.insert("2")
        self.genno_repeat_ngram_size.setValidator(QIntValidator())
        lgenparam.addRow("no_repeat_ngram_size",self.genno_repeat_ngram_size)
        self.gentop_k = QLineEdit()
        self.gentop_k.insert("50")
        self.gentop_k.setValidator(QIntValidator())
        lgenparam.addRow("top_k",self.gentop_k)
        self.gentemperature = QLineEdit()
        self.gentemperature.insert("0.9")
        self.gentemperature.setInputMask("9.99")
        lgenparam.addRow("temperature",self.gentemperature)
        self.genmax_length = QLineEdit()
        self.genmax_length.insert("20")
        self.genmax_length.setValidator(QIntValidator())
        lgenparam.addRow("max_length",self.genmax_length)
        genparam = QWidget()
        genparam.setLayout(lgenparam)
        self.layout.addWidget(genparam) 
        self.layout.addWidget(QLabel('Conversation Files management'))
        self.convhistoriq = QTextEdit()
        self.chemfichier = QLineEdit()
        boutqfiledial = QPushButton('Parcourir...')
        boutqfiledial.clicked.connect(self.parcourir_fichiers)
        bsauver = QPushButton('Save')
        bsauver.clicked.connect(self.sauver_historique)
        bcharger = QPushButton('Load')
        bcharger.clicked.connect(self.charger_historique)
        lignefichier = QHBoxLayout()
        lignefichier.addWidget(self.chemfichier)
        lignefichier.addWidget(boutqfiledial)
        lignefichier.addWidget(bcharger)
        lignefichier.addWidget(bsauver)
        barfichier = QWidget()
        barfichier.setLayout(lignefichier)
        self.layout.addWidget(barfichier)
        self.layout.addWidget(QLabel("Conversation and history"))
        self.laconversation = QTextEdit()
        self.layout.addWidget(self.laconversation)
        self.layout.addWidget(QLabel("Message"))
        lmes = QHBoxLayout()
        self.message = QLineEdit()
        lmes.addWidget(self.message)
        benvoyer = QPushButton('Send')
        lmes.addWidget(benvoyer)
        benvoyer.clicked.connect(self.envoyer)
        wmes = QWidget()
        wmes.setLayout(lmes)
        self.layout.addWidget(wmes)
        self.widj = QWidget()
        self.widj.setLayout(self.layout)

    def gui(self):
        return self.widj

    def tcharger_resogen(self):
        bidule=self.genresoname.text().split(":")
        letyp=bidule[0]
        lename=bidule[1]
        self.resogen=HappyGeneration(letyp,lename)
        loader.close()

    def charger_resogen(self):
        taction = threading.Thread(target=self.tcharger_resogen)
        taction.start()
        loader.exec_()

    def parcourir_fichiers(self):
        dial = QFileDialog()
        dial.setFileMode(QFileDialog.FileMode.AnyFile)
        dial.setNameFilter("text (*.txt)")
        if dial.exec_():
            self.chemfichier.clear()
            self.chemfichier.insert(dial.selectedFiles()[0])

    def tenvoyer(self):
        args = GENSettings(no_repeat_ngram_size=int(self.genno_repeat_ngram_size.text()), do_sample=True, early_stopping=False, top_k=int(self.gentop_k.text()), temperature=float(self.gentemperature.text()),max_length=int(self.genmax_length.text()))
        self.genresult = self.resogen.generate_text(self.laconversation.toPlainText()+"\nIA: ", args=args)
        print(self.genresult.text)
        loader.close()

    def envoyer(self):
        traction = threading.Thread(target=self.tenvoyer)
        self.laconversation.append("ME: "+self.message.text())
        self.message.clear()
        traction.start()
        loader.exec_()
        i=0
        for i in range(len(self.genresult.text)): 
            if self.genresult.text[i]!=' ' and self.genresult.text[i]!="\n" and self.genresult.text[i]!='_':
                break
        dd=self.genresult.text[i:]
        dd=dd.split("\n")[0]
        self.laconversation.append("IA: "+dd)
    
    def sauver_historique(self):
        if self.chemfichier.text()!="":
            f = open(self.chemfichier.text(),"w")
            f.write(self.laconversation.toPlainText())
            f.close()

    def charger_historique(self):
        if os.path.isfile(self.chemfichier.text()):
            self.laconversation.clear()
            with open(self.chemfichier.text(),'r') as f:
                self.laconversation.append(f.read())
            f.close()
###################### QUESTION  ############################
class Question:
    def __init__(self):
        self.resogen=HappyQuestionAnswering()
        self.genresult=[]
        self.ladata=""
        self.generation_gui()

    def generation_gui(self):
        self.layout = QVBoxLayout()
        self.layout.addWidget(QLabel('Network Model Choice'))
        lgenreso=QFormLayout()
        self.genresoname = QLineEdit()
        genresos=["rien"]
        ee=""
        with open('quest.conf','r') as f:
            ee+=f.read()
        f.close()
        genresos = ee.split("\n")
        self.genresoname.setCompleter(QCompleter(genresos,self.genresoname))
        lgenreso.addRow("Network Name",self.genresoname)
        genresocharge = QPushButton("Take it!")
        genresocharge.clicked.connect(self.charger_resogen)
        lgenreso.addRow("Download and load",genresocharge)
        wgenreso = QWidget()
        wgenreso.setLayout(lgenreso)
        self.layout.addWidget(wgenreso)
        self.layout.addWidget(QLabel('Generation Parameters'))
        lgenparam = QFormLayout()
        self.gentop_k = QLineEdit()
        self.gentop_k.insert("50")
        self.gentop_k.setValidator(QIntValidator())
        lgenparam.addRow("Answere number",self.gentop_k)
        genparam = QWidget()
        genparam.setLayout(lgenparam)
        self.layout.addWidget(genparam) 
        self.layout.addWidget(QLabel("Question"))
        lmes = QHBoxLayout()
        self.message = QLineEdit()
        lmes.addWidget(self.message)
        benvoyer = QPushButton('Ask')
        lmes.addWidget(benvoyer)
        benvoyer.clicked.connect(self.envoyer)
        wmes = QWidget()
        wmes.setLayout(lmes)
        self.layout.addWidget(wmes)
        self.layout.addWidget(QLabel('Datafile selection'))
        self.convhistoriq = QTextEdit()
        self.chemfichier = QLineEdit()
        boutqfiledial = QPushButton('Parcourir...')
        boutqfiledial.clicked.connect(self.parcourir_fichiers)
        lignefichier = QHBoxLayout()
        lignefichier.addWidget(self.chemfichier)
        lignefichier.addWidget(boutqfiledial)
        barfichier = QWidget()
        barfichier.setLayout(lignefichier)
        self.layout.addWidget(barfichier)
        self.layout.addWidget(QLabel("Answere"))
        self.reponses = QTableWidget()
        self.reponses.setColumnCount(4)
        self.reponses.setHorizontalHeaderLabels(["score","answer","begin","end"])
        self.layout.addWidget(self.reponses)
        self.widj = QWidget()
        self.widj.setLayout(self.layout)

    def gui(self):
        return self.widj

    def tcharger_resogen(self):
        bidule=self.genresoname.text().split(":")
        letyp=bidule[0]
        lename=bidule[1]
        self.resogen=HappyQuestionAnswering(letyp,lename)
        loader.close()

    def charger_resogen(self):
        taction = threading.Thread(target=self.tcharger_resogen)
        taction.start()
        loader.exec_()

    def parcourir_fichiers(self):
        dial = QFileDialog()
        dial.setFileMode(QFileDialog.FileMode.AnyFile)
        dial.setNameFilter("text (*.txt)")
        if dial.exec_():
            self.chemfichier.clear()
            self.chemfichier.insert(dial.selectedFiles()[0])
            self.charger_data()

    def tenvoyer(self):
        self.genresult = self.resogen.answer_question(self.ladata,self.message.text(),int(self.gentop_k.text()))
        loader.close()

    def envoyer(self):
        traction = threading.Thread(target=self.tenvoyer)
        traction.start()
        loader.exec_()
        self.reponses.clear()
        self.reponses.setHorizontalHeaderLabels(["score","answer","begin","end"])
        self.reponses.setColumnWidth(0,50)
        self.reponses.setColumnWidth(0,250)
        self.reponses.setColumnWidth(0,50)
        self.reponses.setColumnWidth(0,50)
        self.reponses.setRowCount(int(self.gentop_k.text()))
        for i in range(len(self.genresult)):
            print(self.genresult[i])
            self.reponses.setItem(i,0,QTableWidgetItem(str(self.genresult[i].score)))
            self.reponses.setItem(i,1,QTableWidgetItem(str(self.genresult[i].answer)))
            self.reponses.setItem(i,2,QTableWidgetItem(str(self.genresult[i].start)))
            self.reponses.setItem(i,3,QTableWidgetItem(str(self.genresult[i].end)))
    
    def charger_data(self):
        if os.path.isfile(self.chemfichier.text()):
            self.ladata=""
            with open(self.chemfichier.text(),'r') as f:
                self.ladata+=f.read()
            f.close()
################## text2text #######################
class TextGenText:
    def __init__(self):
        self.resogen=HappyTextToText()
        self.genresult=TextToTextResult(text='coucou')
        self.generation_gui()

    def generation_gui(self):
        lgeneration = QVBoxLayout()
        lgeneration.addWidget(QLabel("Network model choice"))
        lgenreso=QFormLayout()
        self.genresoname = QLineEdit()
        genresos=["rien"]
        ee=""
        with open('text2text.conf','r') as f:
            ee+=f.read()
        f.close()
        genresos=ee.split("\n")
        self.genresoname.setCompleter(QCompleter(genresos,self.genresoname))
        lgenreso.addRow("Network Name",self.genresoname)
        genresocharge = QPushButton("Take it!")
        genresocharge.clicked.connect(self.charger_resogen)
        lgenreso.addRow("Download and charge network",genresocharge)
        genreso = QWidget()
        genreso.setLayout(lgenreso)
        lgeneration.addWidget(genreso)
        lgeneration.addWidget(QLabel("Generation parameters"))
        lgenparam = QFormLayout()
        self.top_k = QLineEdit()
        self.top_k.insert("0")
        self.top_k.setValidator(QIntValidator())
        lgenparam.addRow("top_k",self.top_k)
        self.min_length = QLineEdit()
        self.min_length.insert("20")
        self.min_length.setValidator(QIntValidator())
        lgenparam.addRow("min_length",self.min_length)
        self.max_length = QLineEdit()
        self.max_length.insert("20")
        self.max_length.setValidator(QIntValidator())
        lgenparam.addRow("max_length",self.max_length)
        self.temperature = QLineEdit()
        self.temperature.insert("0.7")
        self.temperature.setInputMask("9.99")
        lgenparam.addRow("temperature",self.temperature)
        self.top_p = QLineEdit()
        self.top_p.insert("0.8")
        self.top_p.setInputMask("9.99")
        lgenparam.addRow("top_p",self.top_p)
        genparam = QWidget()
        genparam.setLayout(lgenparam)
        lgeneration.addWidget(genparam)
        lgeneration.addWidget(QLabel("Input text"))
        self.geninput = QTextEdit()
        lgeneration.addWidget(self.geninput)
        bgen = QPushButton("Generate")
        bgen.clicked.connect(self.gen_go)
        lgeneration.addWidget(bgen)
        lgeneration.addWidget(QLabel("Output text"))
        self.genoutput = QTextEdit()
        self.genoutput.setReadOnly(True)
        lgeneration.addWidget(self.genoutput)
        self.fgeneration = QWidget()
        self.fgeneration.setLayout(lgeneration)

    def gui(self):
        return self.fgeneration

    def tcharger_resogen(self):
        bidule=self.genresoname.text().split(":")
        letyp=bidule[0]
        lename=bidule[1]
        print(letyp)
        print(lename)
        self.resogen=HappyTextToText(letyp,lename)
        loader.close()

    def charger_resogen(self):
        taction = threading.Thread(target=self.tcharger_resogen)
        taction.start()
        loader.exec_()

    def tgen_go(self):
        argsa = TTSettings(do_sample=True, top_k=int(self.top_k.text()), top_p=float(self.top_p.text()),temperature=float(self.temperature.text()),min_length=int(self.min_length.text()),max_length=int(self.max_length.text()), early_stopping=True)
        self.genresult = self.resogen.generate_text(self.geninput.toPlainText(), args=argsa)
        print(self.genresult.text)
        loader.close()

    def gen_go(self):
        traction = threading.Thread(target=self.tgen_go)
        traction.start()
        loader.exec_()
        self.genoutput.clear()
        self.genoutput.append(self.genresult.text)

#################################### GUI ##############################################
gentext = GenText()
conversation = Conversation()
question = Question()
text2text = TextGenText()
################################################
fconversation = QWidget()
################################################
fquestion = QWidget()
################################################
ftransformation = QWidget()
################################################
intercalaires = QHBoxLayout()
bgeneration = QPushButton('Generation')
bgeneration.clicked.connect(fct_int_gen)
intercalaires.addWidget(bgeneration)
bconversation = QPushButton('Conversation')
bconversation.clicked.connect(fct_int_conv)
intercalaires.addWidget(bconversation)
bquestion = QPushButton('Question')
bquestion.clicked.connect(fct_int_quest)
intercalaires.addWidget(bquestion)
btransformation = QPushButton('Transformation')
btransformation.clicked.connect(fct_int_tran)
intercalaires.addWidget(btransformation)
gg = QWidget()
gg.setLayout(intercalaires)
fond = QStackedLayout()
fond.addWidget(gentext.gui())
fond.addWidget(conversation.gui())
fond.addWidget(question.gui())
fond.addWidget(text2text.gui())
fond.setCurrentIndex(0)
gg2 = QWidget()
gg2.setLayout(fond)

layout = QVBoxLayout()
layout.addWidget(gg)
layout.addWidget(gg2)
###################################### start ######################################
window.setLayout(layout)
window.show()
sys.exit(app.exec_())


