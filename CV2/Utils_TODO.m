classdef Utils    
    
    properties
        GT
    end
    
    methods
        function   U = Utils(GroundTruth)
         U.GT=GroundTruth;
        end
            
        function c=covert2logical(U,Y,label)
            c=Y;  
        end
        
        function acc=accuracyMC(U,Y)
           U.GT
           acc=0;
        end
        
        function fp=falsePositives(U,Y)
            fp=0;
        end
    
        function fn=falseNegatives(U,Y)
            U.falsePositives(Y)
            fn=0;
        end
        function acc=accuracy(U,Y)
            acc=0;
        end
        function r=recall(U,Y)
            r=0;
        end
        function p=precision(U,Y)
            fn = U.falseNegatives(Y)
            p=0;
        end
        
        function fnr=falseNegativeRate(U,Y)
            fnr=0;
        end
        function fpr=falsePositiveRate(U,Y)
            fpr=0;
        end
        function tp=truePositives(U,Y)
            tp=0;
        end
         
        function tpr=truePositiveRate(U,Y)
            tpr=0;
        end
        
    end
end 
