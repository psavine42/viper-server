
###


#### Process


    REVIT -->  SystemFactory --> System 
                                   |
                                   |
                                  \|/
                                  
                                  
                                  






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


    
                            
"""



            