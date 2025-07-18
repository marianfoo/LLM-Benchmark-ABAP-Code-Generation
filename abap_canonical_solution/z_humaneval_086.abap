CLASS z_humaneval_086 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS anti_shuffle
      IMPORTING
        VALUE(s)      TYPE string
      RETURNING
        VALUE(result) TYPE string.

  PRIVATE SECTION.
ENDCLASS.

CLASS z_humaneval_086 IMPLEMENTATION.

  METHOD anti_shuffle.
    DATA: lt_words  TYPE TABLE OF string,
          lt_sorted TYPE TABLE OF string,
          lt_chars  TYPE TABLE OF string,
          lv_word   TYPE string,
          lv_sorted TYPE string,
          lv_char   TYPE string,
          lv_index  TYPE i,
          lv_len    TYPE i.

    SPLIT s AT ' ' INTO TABLE lt_words.

    LOOP AT lt_words INTO lv_word.
      IF lv_word IS NOT INITIAL.
        CLEAR: lt_chars, lv_sorted.

        " Convert word to character table
        lv_len = strlen( lv_word ).
        DO lv_len TIMES.
          lv_index = sy-index - 1.
          lv_char = lv_word+lv_index(1).
          APPEND lv_char TO lt_chars.
        ENDDO.

        " Sort the characters
        SORT lt_chars.

        " Concatenate sorted characters back to string
        LOOP AT lt_chars INTO lv_char.
          CONCATENATE lv_sorted lv_char INTO lv_sorted.
        ENDLOOP.

        APPEND lv_sorted TO lt_sorted.
      ELSE.
        APPEND '' TO lt_sorted.
      ENDIF.
    ENDLOOP.

    " Join words back with spaces
    LOOP AT lt_sorted INTO lv_word.
      IF sy-tabix = 1.
        result = lv_word.
      ELSE.
        CONCATENATE result lv_word INTO result SEPARATED BY ' '.
      ENDIF.
    ENDLOOP.

  ENDMETHOD.

ENDCLASS.