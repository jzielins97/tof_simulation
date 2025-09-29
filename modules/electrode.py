class TElectrode:
    name:str
    position:[float] = [0,0] # in mm
    length:float = 30.0 # mm
    potential:float = 0 # V
    radius:float = 15 # mm

    def __init__(self, name:str, position:float,length:float):
        self.name = name
        self.position = [position,position+length]
        self.length = length


    def __str__(self)->str:
        return f'Electrode {self.name:5} with {self.potential:4.2e} V from {self.position[0]:5.2f} to {self.position[1]:5.2f}'
    
    def GetName(self):
        return self.name
    
    def GetElectrodeStart(self)->float:
        return self.position[0]
    
    def GetElectrodeCenter(self)->float:
        return (self.position[1] + self.position[0])/2.0
    
    def GetElectrodeEnd(self):
        return self.position[1]
    
    def GetPotential(self)->float:
        return self.potential #  / np.sqrt((self.GetElectrodeCenter() - position)**2 + self.radius**2)

    def SetPotential(self,potential)->None:
        self.potential = potential