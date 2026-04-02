// vocab.h — Vocabulary for Parakeet TDT models (header-only)
#pragma once

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// V2 Vocabulary (1025 BPE tokens, blank=1024)
// From istupakov/parakeet-tdt-0.6b-v2-onnx on HuggingFace (vocab.txt).
// The U+2581 separator is pre-converted to ASCII space.
// ---------------------------------------------------------------------------

static const char* const VOCAB_V2[] = {
    "<unk>"," t"," th"," a","in"," the","re"," w"," o"," s","at","ou","er","nd"," i"," b",
    " c","on"," h","ing"," to"," m","en"," f"," p","an"," d","es","or","ll"," of"," and",
    " y"," l"," I","it"," in","is","ed"," g"," you","ar"," that","om","as"," n","ve",
    "us","ic","ow","al"," it"," be"," wh","le","ion","ut","ot"," we"," is"," e","et",
    "ay"," re"," on"," T"," A"," ha","ent","ke","ct"," S","ig","ver"," Th","all","id",
    " for","ro"," he","se"," this","ld","ly"," go"," k"," st","st","ch"," li"," u","am",
    "ur","ce","ith","im"," so"," have"," do","ht","th"," an"," with","ad"," r","ir",
    " was"," as"," W"," are","ust","ally"," j"," se","ation","od","ere"," like"," not",
    " kn","ight"," B"," they"," And"," know","ome","op"," can"," or"," sh"," me","ill",
    "ant","ck"," what"," at"," ab","ould","ol"," So"," C","use","ter","il"," but",
    " just"," ne"," de","ra","ore"," there","ul","out"," con"," all"," The","ers"," H",
    " fr"," pro","ge","ea"," Y"," O"," M","pp"," com","ess"," ch"," al","est","ate","qu",
    " lo"," ex","very"," su","ain"," one","ca","art","ist","if","ive"," if","ink","nt",
    "ab"," about"," going"," v"," wor","um","ok"," your"," my","ind"," get","cause",
    " from"," don","ri","pe","un","ity"," up"," P"," out","ort"," L","ment","el"," N",
    " some","ich","and"," think","em","oug"," G","os"," D","res"," because"," by","ake",
    " int","ie"," us"," tr"," then","ack"," pl"," here"," pe","her"," will"," F",
    " which","ard"," right"," thing"," want","ies","ople"," It"," them","ame"," We",
    "our"," say"," R"," people"," see"," who","ast","ure","ect","ear"," tim"," E"," You",
    " would"," when","ven"," our","ci"," really"," more","ound","ose","ak"," co","ide",
    "ough"," had","so"," qu","eah"," were","ine"," act","ther"," these"," how"," now",
    " sa","ud"," Wh"," man","ous","one","pt","ff","ong"," has"," any"," very"," But",
    " look","iv","itt"," time"," mo"," ar","hing"," le"," work"," their","are"," his",
    "per","ions"," im"," ag"," J"," no"," en"," got","ag"," sp","ans","act"," te",
    " also","iz","ice"," That"," cl"," been"," way"," fe"," did","ple","ually"," other",
    " U","ite","age","omet","ber","reat","ree"," into","own"," tw"," part","alk",
    " where"," need"," every","pl"," ad","ry"," over","ble","ap","ue"," kind"," po",
    " back"," cont","iff"," somet"," pr","nder","ire"," good"," than","ace"," gu","ep",
    "og","ick","way"," lot"," un"," things"," In","ish","kay"," well"," could"," pre",
    " two","irst"," diff","ach","cc","ittle","int"," He"," those","ence","ip","ase",
    " him"," make"," little","ical"," gr"," year","ass"," thr","uch","ated"," This",
    " off"," res","ac","ance"," actually"," talk","ult","able","orm"," dis"," first",
    "ations"," something"," she","sel"," let","ord"," may","ia"," am"," her"," said",
    " bo","be","ount"," much"," per"," even"," differe","vel","ary"," app","ving",
    " comm"," imp","ys"," again","ress"," yeah"," down","ang"," mean","na","ens"," does",
    " fo"," comp"," ro"," bl","ody"," K"," through"," start","uct"," only"," bet",
    " under"," br"," take","ning"," bu"," use"," Ch","xt","co","ory","ild"," put",
    " call"," new","other","ting"," happ","ater"," inc","ition"," different"," should",
    "ade","ign","thing"," day","fore"," Yeah","ark","ile","ial"," come"," They"," being",
    " try","ious"," sc"," bit"," spe","ub","fe"," doing"," St","vers","av","ty","ian",
    "onna","red","wn"," ke","form","ors"," fl","fter","ail","ents"," gonna"," point",
    "ces"," There","self"," many"," If"," same"," sy"," quest"," most"," great"," What",
    " fu","ug"," show","we","ual","ons"," Be","ically"," ser"," rem"," ind"," pers"," V",
    "he"," str","ved"," still","ank"," rec"," wr","ought","day","ath"," end"," bas","ft",
    "erm","body","ph","ject","ict"," play"," Is","ates"," ph","oth"," acc","get",
    " years"," em"," id"," Oh","ves","ever"," inter"," rel"," before"," feel","igh",
    " three","iss"," des","ne"," why"," uh"," To"," cons"," hel"," after","ower","urn",
    " okay"," long"," bel"," around","ful","te","ise"," ob"," supp","ady","ange","aking",
    " pos","atch"," tra","gr"," might","ert"," help","ost"," too","cial"," world",
    " give","ike"," Okay","ways"," min","ward","ily"," gen"," find"," dec","ular","ob",
    " tell"," Now"," sm"," cour"," real","cess","nds"," big"," num","ction"," add",
    " set"," um","ood","ible"," own"," life","ities"," its"," God","pect"," didn","stem",
    "les","uc","ib","ating","olog"," person"," inv","ably"," sure"," reg","lic"," stu",
    " cr"," ev","ments"," another"," la"," last"," sub"," att"," op"," inst"," sl",
    " happen"," rep"," import","ific","ix"," made"," ear"," ac"," def","ute"," next",
    "ative"," form"," guys"," system","ew"," able","ied"," always","ren","erest"," As",
    " mod"," done","ings"," love","ism"," ask","old","ered"," trans"," count","ility",
    " high"," fin"," prob"," pol"," exam"," pres"," maybe","ell"," stud"," prod"," car",
    "ock"," used","oy","stand"," No"," mon","ks"," interest"," ent","ited"," sort",
    " For"," today","ics"," vide"," bec"," Well"," Al"," important"," such"," run",
    " keep"," fact","ata","ss"," never","ween"," stuff","ract"," question","als"," sim",
    "vern","ather"," course"," Of","oc","ness","arch","ize"," All","ense","blem",
    " probably","hip"," number","ention"," saying"," commun"," An","akes"," belie",
    " between"," better","cus"," place"," gener"," ca"," ins"," ass","cond","cept","ull",
    " understand"," fun"," thought","gan","iew","cy","ution","ope","ason"," problem",
    " doesn","ational"," read"," trying"," sch"," el","ah","atter"," exper"," four",
    " ele"," cou","ont"," called"," partic"," open"," gl"," everything"," eff",
    " getting"," ty"," Am"," Because","ave"," met"," Like","oney"," ","e","t","o","a",
    "n","i","s","h","r","l","d","u","c","y","m","g","w","f","p",",","b",".","k","v","'",
    "I","T","A","S","j","x","W","B","C","?","0","O","-","M","H","Y","q","1","P","z","L",
    "D","N","G","F","R","E","2","J","U",":","5","9","3","K","4","V","8","6","7","!","%",
    "Q","$","Z","X","\xc3\xa9","/","\xc3\xad","\xc3\xa1","\xc2\xa3","\xc3\xb3",
    "\xc4\x81","\xc3\xbc","\xc3\xb1","\xc3\xb6","\xc3\xa8","\xc3\xa7","\xc3\xa0",
    "\xc2\xbf","\xce\xbc","\xcf\x80","\xc3\xa4","\xc3\xba","\xce\xb8","\xc3\xa3",
    "\xcf\x86","\xc4\xab","\xcf\x83","\xc3\xaa","\xcf\x81","\xc3\xa2","\xc3\xb4",
    "^","\xe2\x82\xac","\xc3\x89","\xc5\xab","\xce\x94","\xce\xbb","\xce\xb1",
    "\xcf\x84","\xc3\xa6","\xd0\xb0","\xd0\xbe","\xce\xbd","\xc3\xae","\xce\xb3",
    "\xcf\x88","\xc4\x93","\xd1\x82","\xc3\x9f","\xcf\x89","\xc3\xaf","\xc4\x87",
    "\xc4\x8d","\xce\xb5","\xd0\xb5","\xd0\xb8","\xc3\xb2","\xd1\x80","\xce\xb2",
    "\xc3\xb8","\xc5\x82","\xce\xb4","\xce\xb7","\xd0\xbf","\xc3\xab","\xd0\xbd",
    "\xd1\x81","\xc5\xa1","\xc3\x9c","\xc3\xa5","\xc5\x84","\xc5\x9b","\xd1\x8f",
    "\xc4\x91","\xd0\xbb","\xd0\xbc","\xc3\x96","\xc3\xbb","\xc8\x99","\xd0\xb2",
    "\xc3\x81","\xc3\x98","\xc3\xb9","\xce\xbf","\xd1\x87","\xd1\x8c","\xc5\xbe",
    "\xce\xa6","\xd1\x83","\xc4\x99","\xce\xb9","\xd0\xb1","\xd0\xb3","\xd0\xba",
    "\xc5\x91","\xc5\x9a","\xce\xa9","\xce\xba","\xcf\x85","\xc3\xac","\xc4\x8c",
    "\xce\xad","\xd1\x85","\xd1\x8b","\xc3\x85","\xc3\x87","\xc5\xbc","\xce\xaf",
    "\xce\xb6","\xcf\x87","\xd1\x8d","\xc3\x86","\xc3\x8d","\xc3\xb5","\xc4\x9b",
    "\xc4\xa7","\xc5\x81","\xc5\x93","\xc5\xbd","\xc8\x9b","\xce\x93","\xd0\x9f",
    "\xd0\xb4","\xd0\xb7","\xd1\x84","\xc2\xa1","\xc3\x80","\xc3\x8e","\xc4\x80",
    "\xc4\x97","\xc5\xa0","\xc5\xba","\xce\x9a","\xce\xa8","\xce\xac","\xce\xbe",
    "\xce\xbf","<blk>"
};
// ---------------------------------------------------------------------------
// V3 Vocabulary (8193 multilingual BPE tokens, blank=8192)
// From istupakov/parakeet-tdt-0.6b-v3-onnx on HuggingFace (vocab.txt).
// The U+2581 separator is pre-converted to ASCII space.
// ---------------------------------------------------------------------------
#include "vocab_v3.h"

// Detokenize: join token strings, trim leading space.
// vocab_arr/vocab_size are selected at runtime based on model version.
static std::string detokenize(const std::vector<int>& ids,
                              const char* const* vocab_arr, int vocab_size) {
    std::string text;
    for (int id : ids)
        if (id >= 0 && id < vocab_size)
            text += vocab_arr[id];
    size_t start = text.find_first_not_of(' ');
    return (start == std::string::npos) ? "" : text.substr(start);
}

// Word with start timestamp (ms)
struct WordTiming {
    std::string word;
    int start_ms;
};

// Group BPE tokens into words using leading-space word boundaries.
// token_ids[i] and token_ms[i] are parallel arrays.
static std::vector<WordTiming> words_with_timestamps(
    const std::vector<int>& token_ids,
    const std::vector<int>& token_ms,
    const char* const* vocab_arr, int vocab_size)
{
    std::vector<WordTiming> words;
    std::string current_word;
    int word_start = 0;

    for (size_t i = 0; i < token_ids.size(); i++) {
        int id = token_ids[i];
        if (id < 0 || id >= vocab_size) continue;
        const char* piece = vocab_arr[id];
        bool new_word = (piece[0] == ' ');

        if (new_word && !current_word.empty()) {
            words.push_back({current_word, word_start});
            current_word.clear();
        }

        if (current_word.empty())
            word_start = token_ms[i];

        current_word += new_word ? piece + 1 : piece;
    }

    if (!current_word.empty())
        words.push_back({current_word, word_start});

    return words;
}
