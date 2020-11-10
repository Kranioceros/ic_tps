#Debuguea 'str' si el nivel de debug de 'str' (lvlStr) 
# es mayor al nivel de debug actual en el sistema (lvlActual)
#Con un nivel 0 ves todos los debug
#Con un nuvel 10 ves solo ves debugs mas generales
def dbg(s_str, lvlStr, lvlActual):
    if(lvlActual < 0 or lvlStr < lvlActual):
        return
    
    print(s_str)
