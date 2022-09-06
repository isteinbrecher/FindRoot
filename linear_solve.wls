#!/usr/bin/env wolframscript
(* ::Package:: *)

stringReplace[string_]:=StringReplace[string,{"bb("->"GetV(b,","aa("->"GetM(A,"}]


solve[n_]:=(
A=Table[aa[i,j],{i,0,n-1},{j,0,n-1}];
b=Table[bb[i],{i,0,n-1}];
det=Simplify[Det[A]];
solutionTimesDet=Simplify[LinearSolve[A,b]*det];

outString="";

outString=outString<>"det = "<>stringReplace[ToString[CForm[det]]]<>";\n";

Do[
outString=outString<>"GetV(x,"<>ToString[i-1]<>") = "<>stringReplace[ToString[CForm[solutionTimesDet[[i]]]]]<>";\n";
,{i,n}];

Print[outString];
)

solve[1]
solve[2]
solve[3]
solve[4]
