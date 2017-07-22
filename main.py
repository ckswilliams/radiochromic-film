

from kivy.config import Config
Config.set('kivy', 'log_level', 'debug')

from kivy.app import App
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty
from kivy.properties import StringProperty
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput

from kivy.factory import Factory

from kivy.uix.filechooser import FileChooserListView

from os.path import isdir,join


import numpy as np
import pandas as pd


import film_process as fp
    


def ragged_csv(filename):
    f=open(filename)
    max_n=0
    for line in f.readlines():
        words = len(line.split(','))
        if words > max_n:
            max_n=words
    lines=pd.read_csv(filename,sep=',',names=range(max_n))
    lines[lines!=lines] = ''
    return lines
t=ragged_csv('template/index.txt')


#Main class
class Root(FloatLayout):
    
    directory_input = ObjectProperty(None)
    measurement_cont = ObjectProperty(None)
    measurement_type_input = ObjectProperty(None)
    batch_name_input = ObjectProperty(None)
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    text_input = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super(Root, self).__init__(**kwargs)
        self.measurement_type = 'Measurement type'
        self.batch_name = ''
        self.directory = ''
        #self.load('',['C:/Users/CwPc/Google Drive/msProj/macros/film_process/tst/template'])   
        

    
        
    def dismiss_popup(self):
        self._popup.dismiss()
        
    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Choose directory", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()
        
    def load(self, path, fn):
#        with open(os.path.join(path, filename[0])) as stream:
#            self.text_input.text = stream.read()
        self.directory = fn[0]
        self.load_index()

        return
        self.load_mapindex()

        

                    
    
    def load_index(self):
        print('trying to load')
        try:
            index = ragged_csv(self.directory+'/index.txt')
            header = index[0:1]
            self.measurement_type = header[0][0]
            self.batch_name = header[1][0]
            self.index_data = index
            print(self.index_data)
            
        except FileNotFoundError:
            self.index_data = None
            index = None
            print('No index file, proceeding with blank slate')

            
            
            
#            self.update_index_data()
            
            
        self.update_gui()
        self.display_index()
            


#    def update_index_data(self,data):
#
#        print('working')
#        dat = {}
#        
#        for i in range(len(data)):
#            d = []
#            d.append(data[i][0])
#            if self.measurement_type=='measurement':
#                d.append(data[i][1])
#                d.append(data[i][2:])
#            else:
#                d.append(data[i][1:])
#            dat[i] = d
#        self.index_data = dat
#        self.max_strips = data.shape[1]
#        print(dat)
#        self.update_measurement_content()


    def display_index(self):
        self.measurement_cont.clear_widgets()
        if self.measurement_type == 'measurement':
            self.measurement_cont.add_widget(mRow())
        else:
            self.measurement_cont.add_widget(cRow())
            
        d = self.index_data

        for row in d.iterrows():
            index,dat = row
            if index == 0:
                continue
            r = dat.tolist()
            rowbox = BoxLayout(orientation='horizontal')
            num = Label(text=str(index),size_hint_x=.06)
            rowbox.add_widget(num)
            id_input = TextInput(text=r.pop(0),size_hint_x=.12)
            rowbox.add_widget(id_input)
            #If measurement, add cals
            if self.measurement_type =='measurement':
                cal_input = TextInput(text=r.pop(0),size_hint_x=.12)
                rowbox.add_widget(cal_input)
            #Add all strips
            stripbox = BoxLayout(orientation='horizontal',size_hint_x=.7)
            for s in r:
                text = s
                fileinput = TextInput(text=text)
                stripbox.add_widget(fileinput)
            rowbox.add_widget(stripbox)
            self.measurement_cont.add_widget(rowbox)
            
    def edit_measurement_content(self):
        pass
    
    def update_index(self):
        self.measurement_type = self.measurement_type_input.text
        self.batch_name = self.batch_name_input.text
        self.directory = self.directory_input.text
        data = {}
        for child in self.measurement_cont.children:
            try:
                name = child.children[-2].text
                if self.measurement_type=='measurement':
                    cal = child.children[-3].text
                number = int(child.children[-1].text)
                
                s=[]
                for c in child.children[0].children:
                    s.append(c.text)
                    
                s = list(reversed(s))
                if self.measurement_type=='measurement':
                    data[number] = [name,cal]+s
                else:
                    data[number] = [number,name]+s
            except:
                pass
        df = pd.DataFrame.from_dict(data,orient='index')
        df.loc[0] = [self.measurement_type,self.batch_name]+['' for i in range(df.shape[1]-2)]
        self.index_data = df.sort_index()


        
    def film_process(self):
        self.update_index()
        fp.run_measurement(self.directory)



                

                
    def add_row(self):
        self.update_index()
        s = self.index_data.shape
        self.index_data.loc[s[0]+1] = ['' for i in range(s[1])]
        self.display_index()
        
    def add_column(self):
        self.update_index()
        s = self.index_data.shape
        self.index_data[s[1]] = ['' for i in range(s[0])]
        self.display_index()
        
    def save_index(self):
        #Get index from widgets
        self.update_index()
        self.index_data.to_csv(self.directory+'/index.txt',header=False,index=False)
        #save index to disk
        

    def update_mapindex_data(self):
        pass
        
    def update_gui(self):
        self.directory_input.text = self.directory
        self.measurement_type_input.text = self.measurement_type
        self.batch_name_input.text = self.batch_name




        

#Custom widgets
#class DirectoryChooser(BoxLayout):
#    pass
    #def is_dir(self, directory, filename):
    #    return isdir(join(directory, filename))

class LoadDialog(Popup):
    def is_dir(self, directory, filename):
        return isdir(join(directory, filename))
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

class mRow(BoxLayout):
    pass

class cRow(BoxLayout):
    pass

#Build the app!
class filmApp(App):
    pass#Initialise the App, the super will forcibly pass the self argument to all methods.
#    def __init__(self, **kwargs):
#        super(filmApp, self).__init__(**kwargs)
        # Create an instance of the storage and location classes. The code pertaining to gps, geocoding and file storage should go there

        
Factory.register('Root', cls=Root)
Factory.register('LoadDialog', cls=LoadDialog)
#Factory.register('SaveDialog', cls=SaveDialog)


        
    #Can't forget to actually build the main app
#    def build(self):
#        return Root()
#    def on_pause(self):
#        # Here you can save data if needed
#        return True
#    def on_resume(self):
#        # Here you can check if any data needs replacing (usually nothing)
#        pass

if __name__ == '__main__':
    filmApp().run()
    
    
    
    
    
    
    
    
    
    