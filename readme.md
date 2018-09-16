
###


#### Process


    REVIT -->  SystemFactory --> System 
                                   |
                                   |
                                  \|/
                                  
                                  
                                  

https://threejs.org/docs/#api/core/Raycaster 
src.lib.meshcat.src.meshcat.servers.zmqserver



"""

    +----------KB
    |
conditions -> rule firing -> True, False
    |
    +----- 

inputs:
    ISSYMBOL(node)  

Graph Structure:
    
    TotalSuccs(edge)        -> Number of reachable nodes
    ISSPLIT(node)           -> Nsucs(node) > 1, Npreds(node) = 1 
    
    ISMAIN(edge)            -> not ISBRANCH
    ISBRANCH(node, edge)    -> totalsucs(edge) < Totalsucs( ) and ISSPLIT( )
    
    NPreds(node or edge)    ->
    NSuccs(node or edge)    -> 
    ISEND(node or edge)     -> NSuccs == 0
    
    HBranch(node)       -> ISSPLIT and ISSYMBOL
    VBranch(node)       -> ISSPLIT 
    
    DropHEAD    ->     
    VertHEAD    -> 
    HorzHEAD    ->     

Geometric Structure
    DirectionSame(edge1, edge2)
    
    
    
Logic:
    
    Riser   := ISSYMBOL(node)
    Tee     := 


todo 

1) compute vector at each node by iterating through 'own' prop struct
2) Tree prediction 
3) or (features of prev: angle, normdist, relative-dist) 
    - start with random predictions - 
    3.2) get conditions (room enclosure conditions from model?)
    3.3) method of computing using 'room boundary' information
    
                            
"""



            