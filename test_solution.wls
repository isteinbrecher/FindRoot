#!/usr/bin/env wolframscript
(* ::Package:: *)

getUnknowns[nDim_]:=Table[x[i-1],{i,nDim}]
getInitialGuess[nDim_,val_]:=Thread[{getUnknowns[nDim],val}]


stringReplace[string_]:=Module[{out},
out=string;
out=StringReplace[out,"Power(E,"->"std::exp("];
out=StringReplace[out,"Power("->"std::pow("];
out=StringReplace[out,"Sin("->"std::sin("];
out=StringReplace[out,"Cos("->"std::cos("];
out=StringReplace[out,"x("->"GetV(x,"];
out=StringReplace[out,"f("->"GetV(f,"];
out=StringReplace[out,"b("->"GetV(b,"];
out=StringReplace[out,"A("->"GetM(A,"];
out=StringReplace[out,"jac("->"GetM(jac,"];
out
]

cToString[expr_]:=Module[{string},
string=stringReplace[ToString[CForm[N@expr]]];
string<>";"
]

vectorToString[f_,name_]:=Module[{string},
string="";
Do[
string=string<>"\n"<>name<>"("<>ToString[i-1]<>") = "<>cToString[f[[i]]];
,{i,Length@f}];

stringReplace[string]
]
functionToString[f_]:=vectorToString[f,"f"]

matrixToString[f_,name_]:=Module[{string},
string="";
Do[
string=string<>"\n"<>name<>"("<>ToString[i-1]<>","<>ToString[j-1]<>") = "<>cToString[f[[i,j]]];
,{i,Length@f},{j,Length@f}];

stringReplace[string]
]
jacToString[f_]:=matrixToString[f,"jac"]


(* ::Text:: *)
(*Solution for test cases*)


f={
Sin[x[0]]+Exp[x[0]]
};

functionToString[f]
jacToString[D[f,{getUnknowns[1]}]]

FindRoot[f,getInitialGuess[1,0]]


f={
Exp[x[0]-2]-x[1],
x[1]^2-x[0]
};

functionToString[f]
jacToString[D[f,{getUnknowns[2]}]]

FindRoot[f,getInitialGuess[2,0]]


f={
Exp[x[0]-2]-x[1],
x[1]^2-x[2]*x[0],
x[0]+Sin[x[1]]
};

functionToString[f]
jacToString[D[f,{getUnknowns[3]}]]

FindRoot[f,getInitialGuess[3,1]]


SeedRandom[666]

getSolution[n_]:=Module[{A,b,x,string=""},
b=RandomReal[{-1,1},{n}];
A=RandomReal[{-1,1},{n,n}];

string=string<>matrixToString[A,"A"];
string=string<>vectorToString[b,"b"];
string=string<>"\n"<>vectorToString[LinearSolve[A,b],"x"];
string=string<>"\n\ndet = "<>cToString[Det[A]];
string
]

getSolution[1]
getSolution[2]
getSolution[3]
getSolution[4]
