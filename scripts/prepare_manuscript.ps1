$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$sourcePath = Join-Path $repoRoot 'papers\mineral_ontology_paper_en_extraction_task.md'
$bibliographyPath = Join-Path $repoRoot 'papers\references.bib'
$pandocPath = 'D:\anaconda\Scripts\pandoc.exe'

if (-not (Test-Path -LiteralPath $sourcePath)) {
    throw "English manuscript source was not found: $sourcePath"
}
if (-not (Test-Path -LiteralPath $bibliographyPath)) {
    throw "Bibliography source was not found: $bibliographyPath"
}
if (-not (Test-Path -LiteralPath $pandocPath)) {
    throw "Pandoc was not found: $pandocPath"
}

$referenceMap = @{
    1 = 'gruber1995translation'; 2 = 'studer1998knowledge'; 3 = 'sen2007geosciml'
    4 = 'ogcGeosciml41'; 5 = 'tao2025gkere'; 6 = 'liu2025prospectingkg'
    7 = 'zhao2025pcdkg'; 8 = 'babaei2024llms4ol'; 9 = 'fathallah2025neongpt'
    10 = 'shimizu2024accelerating'; 11 = 'chepurova2026wikontic'; 12 = 'li2026ontokg'
    13 = 'li2026lsdts'; 14 = 'salovskii2026automatic'; 15 = 'zeng2025codetaxo'
    16 = 'xue2024autore'; 17 = 'hu2025intervention'; 18 = 'munne2025entity'
    19 = 'li2025arter'; 20 = 'feng2024pipeline'; 21 = 'tian2025pminrkg'
    22 = 'zhao2025llmmineral'; 23 = 'oarga2026scientific'; 24 = 'talukder2026automated'
    25 = 'niu2025treekg'; 26 = 'pan2025taxonomy'; 27 = 'huang2025llmsgraphjudge'
    28 = 'mihindukulasooriya2023text2kgbench'; 29 = 'bian2025llmkgcsurvey'
    30 = 'tang2026chisagent'; 31 = 'lippolis2025ontology'; 32 = 'li2025mathgeosci'
    33 = 'bai2025frameworkkg'; 34 = 'ciatto2025oracles'; 35 = 'marchesin2025dataquality'
    36 = 'gao2025mdkg'; 37 = 'lu2022uie'; 38 = 'lu2021text2event'; 39 = 'zhou2021atlop'
}

function Expand-ReferenceList {
    param([string]$ReferenceList)

    $keys = [System.Collections.Generic.List[string]]::new()
    foreach ($item in ($ReferenceList -split ',')) {
        $part = $item.Trim()
        if ($part -match '^(\d+)\s*-\s*(\d+)$') {
            $start = [int]$Matches[1]
            $end = [int]$Matches[2]
            foreach ($number in $start..$end) {
                if (-not $referenceMap.ContainsKey($number)) {
                    throw "No BibTeX key is mapped for reference [$number]."
                }
                $keys.Add($referenceMap[$number])
            }
        }
        elseif ($part -match '^\d+$') {
            $number = [int]$part
            if (-not $referenceMap.ContainsKey($number)) {
                throw "No BibTeX key is mapped for reference [$number]."
            }
            $keys.Add($referenceMap[$number])
        }
        else {
            throw "Unsupported numeric reference list: [$ReferenceList]"
        }
    }
    return '\citep{' + ($keys -join ',') + '}'
}

function Convert-NumericCitations {
    param([string]$Text)

    $pattern = '\[(?<references>\d+(?:\s*(?:,|-)\s*\d+)*)\]'
    return [regex]::Replace($Text, $pattern, {
            param($match)
            Expand-ReferenceList $match.Groups['references'].Value
        })
}

function Convert-TaggedDisplayMath {
    param([string]$Latex)

    $pattern = '(?s)\\\[\s*(?<math>.*?)\\qquad\\text\{\((?<tag>3-\d+)\)\}\s*\\\]'
    return [regex]::Replace($Latex, $pattern, {
            param($match)
            $math = $match.Groups['math'].Value.Trim()
            $tag = $match.Groups['tag'].Value
            "\begin{equation}`r`n$math`r`n\tag{$tag}`r`n\end{equation}"
        })
}

function Convert-MarkdownToLatex {
    param(
        [string]$Markdown,
        [string]$TargetPath
    )

    $temporaryMarkdown = [System.IO.Path]::GetTempFileName()
    try {
        Set-Content -LiteralPath $temporaryMarkdown -Value $Markdown -Encoding utf8
        & $pandocPath $temporaryMarkdown -f 'markdown+raw_tex' -t latex --wrap=none -o $TargetPath
        if ($LASTEXITCODE -ne 0) {
            throw "Pandoc failed while creating $TargetPath"
        }
        $latex = Get-Content -LiteralPath $TargetPath -Raw -Encoding utf8
        $latex = Convert-TaggedDisplayMath $latex
        # svjour3 predates Pandoc's hyperlink anchors; headings keep their labels.
        $latex = [regex]::Replace($latex, '(?m)^\\hypertarget\{[^}]+\}\{%\r?\n', '')
        $latex = [regex]::Replace($latex, '(\\label\{[^}]+\})\}', '$1')
        # Keep compact method schematics visually secondary to the overview figures.
        $latex = [regex]::Replace($latex, '\\includegraphics(?:\[[^\]]*\])?\{(?<figure>[^}]+)\}', {
                param($match)
                $figure = $match.Groups['figure'].Value
                $width = switch ($figure) {
                    'figures/taxonomy_skeleton_method.png' { '0.72\linewidth' }
                    'figures/relation_rhf_extraction_method_v2.png' { '0.84\linewidth' }
                    default { '\linewidth' }
                }
                '\includegraphics[width=' + $width + ',height=0.78\textheight,keepaspectratio]{' + $figure + '}'
            })
        # The class supplies the figure number; retain only the descriptive caption.
        $latex = [regex]::Replace($latex, '\\caption\{Figure\s+(?:\d+-\d+|\d+)\.\s*', '\caption{')
        $latex = $latex -replace '\\emph\{Mathematical Geosciences\}', 'Mathematical Geosciences'
        # The journal package is uploaded in one directory; keep figure names flat.
        $latex = $latex -replace 'figures/', ''
        Set-Content -LiteralPath $TargetPath -Value $latex -Encoding utf8
    }
    finally {
        Remove-Item -LiteralPath $temporaryMarkdown -Force -ErrorAction SilentlyContinue
    }
}

$source = Get-Content -LiteralPath $sourcePath -Raw -Encoding utf8
$front = [regex]::Match($source, '(?s)^#\s+(?<title>.+?)\r?\n\r?\n##\s+Abstract\r?\n\r?\n(?<abstract>.*?)\r?\n\r?\nKeywords:\s*(?<keywords>[^\r\n]+)\r?\n\r?\n(?<article>#\s+1\s+Introduction.*)$')
if (-not $front.Success) {
    throw 'The expected title, abstract, keywords, and Introduction structure was not found.'
}

$article = $front.Groups['article'].Value
$referencesIndex = $article.IndexOf("`n# References")
if ($referencesIndex -lt 0) {
    throw 'The References heading was not found.'
}

$appendixMarker = "`n# Appendices"
$appendixIndex = $article.IndexOf($appendixMarker, $referencesIndex)
if ($appendixIndex -lt 0) {
    throw 'The Appendices heading was not found.'
}

$main = $article.Substring(0, $referencesIndex).Trim()
$appendix = $article.Substring($appendixIndex + $appendixMarker.Length).Trim()

$main = [regex]::Replace($main, '(?m)^#\s+\d+\s+', '# ')
$main = [regex]::Replace($main, '(?m)^##\s+\d+\.\d+\s+', '## ')
$appendix = [regex]::Replace($appendix, '(?m)^##\s+Appendix\s+[A-Z]\s+', '# ')
$appendix = [regex]::Replace($appendix, '(?m)^###\s+[A-Z]\.\d+\s+', '## ')

$abstract = Convert-NumericCitations $front.Groups['abstract'].Value.Trim()
$main = Convert-NumericCitations $main
$appendix = Convert-NumericCitations $appendix

Set-Content -LiteralPath (Join-Path $PSScriptRoot 'abstract.tex') -Value $abstract -Encoding utf8
Convert-MarkdownToLatex -Markdown $main -TargetPath (Join-Path $PSScriptRoot 'manuscript_body_main.tex')
Convert-MarkdownToLatex -Markdown $appendix -TargetPath (Join-Path $PSScriptRoot 'manuscript_body_appendix.tex')
Copy-Item -LiteralPath $bibliographyPath -Destination (Join-Path $PSScriptRoot 'references.bib') -Force

# Keep the paper-ready figures synchronized with the English manuscript source.
$paperFigures = @(
    'method_overall_pipeline_final.pdf',
    'taxonomy_skeleton_method.png',
    'relation_rhf_extraction_method_v2.png',
    'two_way_logic_validation_method.pdf',
    'term_alignment_rdf_method.png',
    'extracted_ontology_subgraph.png',
    'algorithm_b1_pseudocode_en.png',
    'table_4_1_validation_matrix_en.png',
    'table_4_2_evaluation_metrics_en.png',
    'table_4_3_final50_metrics_en.png',
    'table_4_4_ablation_en.png'
)
foreach ($figureName in $paperFigures) {
    $figurePath = Join-Path $repoRoot (Join-Path 'papers\figures' $figureName)
    if (-not (Test-Path -LiteralPath $figurePath)) {
        throw "Paper figure was not found: $figurePath"
    }
    Copy-Item -LiteralPath $figurePath -Destination (Join-Path $PSScriptRoot $figureName) -Force
}

Write-Host 'Prepared Mathematical Geosciences manuscript body, appendices, and author-year citations.'
