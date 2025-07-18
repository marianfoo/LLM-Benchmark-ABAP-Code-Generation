CLASS z_humaneval_117 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: select_words
      IMPORTING
        s             TYPE string
        n             TYPE i
      RETURNING
        VALUE(result) TYPE string_table.

ENDCLASS.

CLASS z_humaneval_117 IMPLEMENTATION.

  METHOD select_words.
    DATA: words        TYPE string_table,
          word         TYPE string,
          char         TYPE c LENGTH 1,
          n_consonants TYPE i,
          i            TYPE i.

    IF s IS INITIAL.
      RETURN.
    ENDIF.

    SPLIT s AT ' ' INTO TABLE words.

    LOOP AT words INTO word.
      n_consonants = 0.
      DO strlen( word ) TIMES.
        i = sy-index - 1.
        char = word+i(1).
        TRANSLATE char TO LOWER CASE.
        IF char <> 'a' AND char <> 'e' AND char <> 'i' AND char <> 'o' AND char <> 'u'.
          n_consonants = n_consonants + 1.
        ENDIF.
      ENDDO.
      IF n_consonants = n.
        APPEND word TO result.
      ENDIF.
    ENDLOOP.

  ENDMETHOD.

ENDCLASS.