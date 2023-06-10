import os
from unittest import result
import creat_topology as ct
import pandas as pd
def getFileName(path,suffix):
    input_template_All=[]
    input_template_All_Path=[]
    for root, dirs, files in os.walk(path, topdown=False):
         for name in files:
             #print(os.path.join(root, name))
             print(name)
             if os.path.splitext(name)[1] == suffix:
                 input_template_All.append(name)
                 input_template_All_Path.append(os.path.join(root, name))
        
    return input_template_All,input_template_All_Path

path='archive_test/'
 
input_all,input_all_path=getFileName(path,'.gml')
all_net=[]
degree=[]
node_num=[]
link_num=[]
error_list=['archive_test/Airtel.gml','archive_test/Arnes.gml','archive_test/Arpanet19719.gml','archive_test/Arpanet19723.gml','archive_test/Arpanet19728.gml','archive_test/AsnetAm.gml','archive_test/AttMpls.gml','archive_test/Azrena.gml',
'archive_test/Bellcanada.gml','archive_test/Bellsouth.gml','archive_test/Belnet2003.gml','archive_test/Belnet2004.gml','archive_test/Belnet2005.gml','archive_test/Belnet2006.gml','archive_test/Belnet2007.gml'
,'archive_test/Belnet2008.gml','archive_test/Belnet2009.gml','archive_test/Belnet2010.gml','archive_test/BeyondTheNetwork.gml','archive_test/Bren.gml','archive_test/BsonetEurope.gml','archive_test/BtAsiaPac.gml',
'archive_test/BtEurope.gml','archive_test/BtLatinAmerica.gml','archive_test/BtNorthAmerica.gml','archive_test/Canerie.gml','archive_test/Carnet.gml','archive_test/Cernet.gml',
'archive_test/Cesnet1997.gml','archive_test/Cogentco.gml','archive_test/Colt.gml','archive_test/Columbus.gml','archive_test/Compuserve.gml'
,'archive_test/Cudi.gml','archive_test/Cwix.gml','archive_test/Deltacom.gml','archive_test/Dfn.gml','archive_test/DialtelecomCz.gml','archive_test/Digex.gml'
,'archive_test/Easynet.gml','archive_test/Eenet.gml','archive_test/Ernet.gml','archive_test/Esnet.gml','archive_test/Eunetworks.gml',
'archive_test/Fatman.gml','archive_test/Fccn.gml','archive_test/Funet.gml','archive_test/Gambia.gml','archive_test/Garr199904.gml',
'archive_test/Garr199905.gml','archive_test/Garr200109.gml','archive_test/Garr200112.gml','archive_test/Garr201201.gml',
'archive_test/Geant2010.gml','archive_test/Globenet.gml','archive_test/Grena.gml','archive_test/Grnet.gml'
,'archive_test/GtsCe.gml','archive_test/GtsHungary.gml','archive_test/GtsRomania.gml','archive_test/GtsSlovakia.gml'
,'archive_test/Harnet.gml','archive_test/Heanet.gml','archive_test/HiberniaCanada.gml','archive_test/HiberniaGlobal.gml',
'archive_test/HiberniaIreland.gml','archive_test/HiberniaNireland.gml','archive_test/HiberniaUk.gml','archive_test/HiberniaUs.gml'
,'archive_test/Highwinds.gml','archive_test/Iij.gml','archive_test/Iinet.gml','archive_test/Intellifiber.gml'
,'archive_test/Internetmci.gml','archive_test/Internode.gml','archive_test/Interoute.gml','archive_test/Intranetwork.gml'
,'archive_test/Ion.gml','archive_test/IowaStatewideFiberMap.gml','archive_test/Iris.gml','archive_test/Istar.gml'
,'archive_test/Janetlense.gml','archive_test/Karen.gml','archive_test/Kdl.gml','archive_test/KentmanApr2007.gml'
,'archive_test/KentmanFeb2008.gml','archive_test/KentmanJan2011.gml','archive_test/Marwan.gml','archive_test/Missouri.gml'
,'archive_test/Myren.gml','archive_test/Nextgen.gml','archive_test/Nordu2005.gml','archive_test/Nordu2010.gml'
,'archive_test/Nsfcnet.gml','archive_test/Ntelos.gml','archive_test/Ntt.gml','archive_test/Oteglobe.gml'
,'archive_test/Oxford.gml','archive_test/Pacificwave.gml','archive_test/Palmetto.gml','archive_test/Pern.gml'
,'archive_test/PionierL1.gml','archive_test/PionierL3.gml','archive_test/RedBestel.gml','archive_test/Rediris.gml'
,'archive_test/Reuna.gml','archive_test/Roedunet.gml','archive_test/RoedunetFibre.gml','archive_test/Shentel.gml'
,'archive_test/Sunet.gml','archive_test/Surfnet.gml','archive_test/Switch.gml','archive_test/SwitchL3.gml'
,'archive_test/Syringa.gml','archive_test/TataNld.gml','archive_test/Tinet.gml','archive_test/TLex.gml'
,'archive_test/Tw.gml','archive_test/Ulaknet.gml','archive_test/UniC.gml','archive_test/Uninet.gml'
,'archive_test/Uninett2010.gml','archive_test/Uninett2011.gml','archive_test/UsCarrier.gml','archive_test/UsSignal.gml'
,'archive_test/Uunet.gml','archive_test/Vinaren.gml','archive_test/WideJpn.gml','archive_test/Zamren.gml']
for gml_path in input_all_path:
    if gml_path not in error_list:
        d,n,l=ct.read_net_degree(gml_path)
        all_net.append(gml_path)
        degree.append(d)
        node_num.append(n)
        link_num.append(l)
dfdata={
    "name":all_net,
    "degree":degree,
    "nodenum":node_num,
    "linknum":link_num,
}
df = pd.DataFrame(dfdata)
writer = pd.ExcelWriter("archive.xlsx")
df.to_excel(writer, index=False)
writer.close()