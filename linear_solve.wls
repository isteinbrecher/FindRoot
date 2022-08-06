#!/usr/bin/env wolframscript
(* ::Package:: *)

solve[n_]:=(
A=Table[aa[i,j],{i,n},{j,n}];
b=Table[bb[i],{i,n}];
Print[Simplify[LinearSolve[A,b]]];
Print[Simplify[LinearSolve[A,b]*Det[A]]];
Print[Simplify[Det[A]]];
)
solve[1]
solve[2]
solve[3]
solve[4]
solve[5]
