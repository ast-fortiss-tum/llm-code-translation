import os
import regex

# f"./tables_tex/iteration_1_stats_percent_{pair[0]}_{pair[1]}.tex", float_format="%.2f")

all_tables = ""
for file_name in os.listdir("./tables_tex"):
    if "iteration_1_stats_percent_" in file_name:
        name_parts = file_name.split(".")[0].split("_")
        source_pl = name_parts[-2]
        target_pl = name_parts[-1]

        with open("./tables_tex/" + file_name, "r") as f:
            text = f.read()
            text = text.replace(
                r"\multirow[t]{5}{*}{All} &",
                r"""\textbf{All} & & & & & & & & & & & & & & \\ 
\quad""",
            )  # ???
            text = text.replace(
                r"\multirow[t]{5}{*}{codenet} &",
                r"""\textbf{CodeNet} & & & & & & & & & & & & & & \\ 
\quad""",
            )
            text = text.replace(
                r"\multirow[t]{5}{*}{avatar} &",
                r"""\textbf{AVATAR} & & & & & & & & & & & & & & \\
\quad""",
            )
            text = text.replace(
                r"\multirow[t]{5}{*}{bithacks} &",
                r"""\textbf{Bithacks} & & & & & & & & & & & & & & \\ 
\quad""",
            )

            text = text.replace(r"Dataset & result_1 &  &  &  &  &  &  &  &  &  &  &  &  &  &  \\", "")
            text = regex.sub(r"& All & 100\.00 [^\\]*\\\\", "", text)
            if target_pl == "total":
                caption = "Detailed execution metrics for all datasets, prompt templates, and models. The numbers are aggregated for all language pairs."
                label = "tab:iteration_1_stats_percent_total"

            else:
                caption = f"Detailed execution metrics for all datasets, prompt templates, and models for translations from {source_pl} to {target_pl}"
                label = f"tab:iteration_1_stats_percent_{source_pl}_{target_pl}"
            text = text.replace(
                "begin{tabular}{llrrrrrrrrrrrrrr}",
                r"""begin{table}[t]
\caption{"""
                + caption
                + """}
\label{"""
                + label
                + """}

\setlength{\\tabcolsep}{2.5pt} % spacing between cells
\\renewcommand{\\arraystretch}{1} % spacing between rows

\\footnotesize
\\begin{tabular}{@{}lcccccccccccccc@{}}""",
            )

            text = text.replace(
                r"""& Model & Codestral & \multicolumn{3}{r}{D-Mistral} & \multicolumn{2}{r}{D-Phi-2} & D-Mixtral & Llama 3 & \multicolumn{3}{r}{Mistral} & \multicolumn{2}{r}{Mixtral} & Phi-3 \\
 & Prompt & MD & RM & MD & VT & RM & MD & MD & MD & RM & MD & VT & RM & MD & MD \\""",
                r"""& Codestral & \multicolumn{3}{c}{D-Mistral} & \multicolumn{2}{c}{D-Phi-2} & D-Mixtral & Llama 3 & \multicolumn{3}{c}{Mistral}  & \multicolumn{2}{c}{Mixtral} & Phi-3 \\ 

% \midrule
\cmidrule(l){2-2}
\cmidrule(l){3-5}
\cmidrule(l){6-7}
\cmidrule(l){8-8}
\cmidrule(l){9-9}
\cmidrule(l){10-12}
\cmidrule(l){13-14}
\cmidrule(l){15-15}

& \multicolumn{1}{c}{MD}
& \multicolumn{1}{c}{RM}
& \multicolumn{1}{c}{MD}
& \multicolumn{1}{c}{VT}
& \multicolumn{1}{c}{RM}
& \multicolumn{1}{c}{MD} 
& \multicolumn{1}{c}{MD} 
& \multicolumn{1}{c}{MD} 
& \multicolumn{1}{c}{RM} 
& \multicolumn{1}{c}{MD} 
& \multicolumn{1}{c}{VT} 
& \multicolumn{1}{c}{RM}
& \multicolumn{1}{c}{MD} 
& \multicolumn{1}{c}{MD} \\""",
            )

            text = text.replace(
                r"\end{tabular}",
                r"""\end{tabular}
\end{table}""",
            )
            # remove internal hiorizontal lines
            text = regex.sub(r"\\cline{1-\d\d}", "", text)
            text = text.replace(
                """\\\\
 &""",
                r"""\\
\qquad""",
            )
            text = text.replace("C#", "C\#", 1)
            text = text.replace("C#", "Csharp")
            text = text.replace(
                """\end{table}
\end{table}""",
                "\end{table}",
            )
            text = text.replace("100.00", "100.0")

        with open("./tables_tex/" + file_name, "w") as f:
            f.write(text)
            if target_pl == "total":
                all_tables = text + "\n\n\n\n" + all_tables
            else:
                all_tables += text + "\n\n\n\n"

with open("./tables_tex/iteration_1_stats_tables.tex", "w") as f:
    f.write(all_tables)
