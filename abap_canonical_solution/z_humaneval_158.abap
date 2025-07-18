CLASS z_humaneval_158 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: find_max
      IMPORTING
        words         TYPE string_table
      RETURNING
        VALUE(result) TYPE string.

ENDCLASS.

CLASS z_humaneval_158 IMPLEMENTATION.

  METHOD find_max.
    DATA: max_unique_count TYPE i,
          unique_count     TYPE i,
          current_word     TYPE string,
          best_word        TYPE string,
          char_set         TYPE string_table,
          char             TYPE string,
          len              TYPE i,
          i                TYPE i.

    max_unique_count = -1.

    LOOP AT words INTO current_word.
      CLEAR char_set.
      len = strlen( current_word ).

      DO len TIMES.
        i = sy-index - 1.
        char = current_word+i(1).
        READ TABLE char_set TRANSPORTING NO FIELDS WITH KEY table_line = char.
        IF sy-subrc <> 0.
          APPEND char TO char_set.
        ENDIF.
      ENDDO.

      unique_count = lines( char_set ).

      IF unique_count > max_unique_count OR
         ( unique_count = max_unique_count AND current_word < best_word ).
        max_unique_count = unique_count.
        best_word = current_word.
      ENDIF.
    ENDLOOP.

    result = best_word.
  ENDMETHOD.

ENDCLASS.