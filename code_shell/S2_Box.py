########################################################
## Run without GUI with following command 
#     abaqus cae noGUI=AbaRun.py
#########################################################

## Default ABAQUS inputs
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
########################################################
import numpy as np
import M3_AbqFunctions as M3AF   
########################################################
### Parameters
### Load parameters from image
sample_name = 'shell_sample_name'
Dim = np.loadtxt(sample_name+'_ImgDim.txt')
Length = Dim[0] * 1e-3
Thickness = Dim[1] * 1e-3

# Set element size
ElemSize = 70*1e-3

NameModel = sample_name+'_BoxModel'
JobName = 'Job-'+NameModel

x0 = 0.0; x1 = Length; y0 = 0.0 ; y1 = Thickness
Disp = -1.0 * Length * 100 ** (-1)

########################################################
### Define Model Object
modelObj = mdb.Model(name=NameModel)

### Square box part
modelObj=M3AF.AbqBox(modelObj,x0,x1,y0,y1)

### Delete obselete Model-1
del mdb.models['Model-1']

### Define set-names
partObj=modelObj.parts['Part-1']
partObj=M3AF.AbqBoxSet(partObj,x0,x1,y0,y1)

## Material Definition
modelObj.Material(name='Material-1')
modelObj.materials['Material-1'].UserDefinedField()
modelObj.materials['Material-1'].Depvar(n=1)
modelObj.materials['Material-1'].Elastic(dependencies=1, 
    table=((23182.0, 1720.861913, 1720.861913, 0.386, 0.386, 0.4, 614.6420882, 
    614.6420882, 553.5714286, 0.1), (44814.0, 1934.060112, 1934.060112, 0.372, 
    0.372, 0.4, 690.8584112, 690.8584112, 553.5714286, 0.2), (66446.0, 
    2207.554879, 2207.554879, 0.358, 0.358, 0.4, 788.6521858, 788.6521858, 
    553.5714286, 0.3), (88078.0, 2571.138707, 2571.138707, 0.344, 0.344, 0.4, 
    918.697495, 918.697495, 553.5714286, 0.4), (109710.0, 3078.101358, 
    3078.101358, 0.33, 0.33, 0.4, 1100.099032, 1100.099032, 553.5714286, 0.5), 
    (131342.0, 3834.084561, 3834.084561, 0.316, 0.316, 0.4, 1370.763279, 
    1370.763279, 553.5714286, 0.6), (152974.0, 5082.299913, 5082.299913, 0.302, 
    0.302, 0.4, 1818.075857, 1818.075857, 553.5714286, 0.7), (174606.0, 
    7535.558085, 7535.558085, 0.288, 0.288, 0.4, 2698.738133, 2698.738133, 
    553.5714286, 0.8), (196238.0, 14567.27202, 14567.27202, 0.274, 0.274, 0.4, 
    5234.093833, 5234.093833, 553.5714286, 0.9)), type=ENGINEERING_CONSTANTS)

### Section
modelObj.HomogeneousSolidSection(material='Material-1',
                                 name='Section-1', thickness=1.0)

### Section assignment
partObj.SectionAssignment(offset=0.0,offsetField='', offsetType=MIDDLE_SURFACE, 
                          region=partObj.sets['Box'], sectionName='Section-1',
                          thicknessAssignment=FROM_SECTION)

### Material Orientation
partObj.MaterialOrientation(additionalRotationType=ROTATION_NONE, fieldName='', 
                            localCsys=None, orientationType=USER, 
                            region=partObj.sets['Box'], stackDirection=STACK_3)

                            
### Meshing
partObj=M3AF.AbqBoxMesh(partObj,x0,x1,y0,y1,ElemSize)

### Assemply
modelObj.rootAssembly.DatumCsysByDefault(CARTESIAN)
modelObj.rootAssembly.Instance(dependent=ON, name='Part-1-1',part=partObj)
instanceObj=modelObj.rootAssembly.instances['Part-1-1']

### Define static step
modelObj.StaticStep(initialInc=0.1, maxInc=0.1, name='Step-1',
                    nlgeom=ON, previous='Initial')


### BC
modelObj.DisplacementBC(amplitude=UNSET, createStepName='Initial', 
	distributionType=UNIFORM, fieldName='', localCsys=None, name='BC-1', 
    region=instanceObj.sets['x0'], u1=SET, u2=UNSET, u3=UNSET,
    ur1=UNSET,ur2=UNSET,ur3=UNSET)
modelObj.DisplacementBC(amplitude=UNSET, createStepName='Initial', 
	distributionType=UNIFORM, fieldName='', localCsys=None, name='BC-2', 
    region=instanceObj.sets['xy0'], u1=UNSET, u2=SET, u3=UNSET,
    ur1=UNSET,ur2=UNSET,ur3=UNSET)

modelObj.DisplacementBC(amplitude=UNSET, 
    createStepName='Step-1', distributionType=UNIFORM, fieldName='', fixed=OFF, 
    localCsys=None, name='BC-3', region=instanceObj.sets['x1']
    , u1=Disp, u2=UNSET, ur3=UNSET)

### Field Output
modelObj.FieldOutputRequest(createStepName='Step-1', name=
    'F-Output-1', variables=('SDV','FV',))
modelObj.FieldOutputRequest(createStepName='Step-1', name=
    'F-Output-2', variables=('S','E','U','UR'))
    
### History output
modelObj.HistoryOutputRequest(createStepName='Step-1', name=
    'H-Output-1', rebar=EXCLUDE, region=instanceObj.sets['x1'], 
    sectionPoints=DEFAULT, variables=('RF1', 'U1'))
 
##################### Job definition ########################### 
mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF, 
	explicitPrecision=SINGLE, getMemoryFromAnalysis=True, historyPrint=OFF, 
	memory=90, memoryUnits=PERCENTAGE, model=NameModel, modelPrint=OFF, 
	multiprocessingMode=MPI, name=NameModel, nodalOutputPrecision=SINGLE, 
	numCpus=1, numGPUs=0, queue=None, resultsFormat=ODB, scratch='', type=
	ANALYSIS, userSubroutine='orient.f', waitHours=0, waitMinutes=0)

######################## Save the CAE file #######################

mdb.jobs[NameModel].writeInput(consistencyChecking=OFF) 
mdb.saveAs(pathName=NameModel+'.cae')

### Export integration point coordinates
with open(NameModel + '.inp') as f:
    lines = f.readlines()

def find_line(lines, word):
    for line in lines:
        if line.find(word) !=-1:
            return lines.index(line)

W_START = '*Part, name=Part-1'
I_START = find_line(lines, W_START) + 1
W_END = '*Orientation, name=Ori-1, system=user'
I_END = find_line(lines, W_END)

new_lines = lines[I_START : I_END]
new_lines.append('** Section: Section-1\n')
new_lines.append('*Solid Section, elset=Box, material=Default MATERIAL\n')
new_lines.append(',\n')

with open(sample_name+'_IP1.inp', 'w') as f:
    for line in new_lines:
        f.write(line)
    f.close()

IP2 = sample_name+'_IP2'
JOB1 = mdb.JobFromInputFile(name=IP2, inputFileName=IP2+'.inp')
JOB1.submit()
JOB1.waitForCompletion()