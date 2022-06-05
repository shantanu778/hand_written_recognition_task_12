from Levenshtein import distance as levenshtein_distance
import os

def eval(infile, gtFile):

    with open(infile,'r') as file:
        preds = [line.strip() for line in file]
    with open(gtFile,'r') as file:
        gt = [line.strip() for line in file]

    score = 0
    for p,g in zip(preds,gt):
        score += levenshtein_distance(p,g)

    return score

def main():

    resultsDir = 'Text_files'
    gtDir = 'gt_files'

    resultFiles = os.listdir(resultsDir)
    score = 0
    for file in resultFiles:
        predFile = os.path.join(resultsDir,file)
        gtFile = os.path.join(gtDir,file)
        s = eval(predFile,gtFile)
        print(f"{file}: {s}")
        score+= s
    
    print(f"Avg: {score/len(resultFiles)}")



if __name__ == "__main__":

    main()