CLASS z_humaneval_016 DEFINITION
PUBLIC
FINAL
CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS count_distinct_characters
      IMPORTING
        VALUE(iv_string) TYPE string
      RETURNING
        VALUE(rv_count)  TYPE i.

  PRIVATE SECTION.
ENDCLASS.

CLASS z_humaneval_016 IMPLEMENTATION.

  METHOD count_distinct_characters.
    DATA lt_chars TYPE TABLE OF string.
    DATA lv_lower TYPE string.
    DATA lv_char TYPE string.
    DATA lv_len TYPE i.
    DATA lv_index TYPE i.

    lv_lower = to_lower( iv_string ).
    lv_len = strlen( lv_lower ).

    DO lv_len TIMES.
      lv_index = sy-index - 1.
      lv_char = lv_lower+lv_index(1).

      IF lv_char IS NOT INITIAL.
        READ TABLE lt_chars TRANSPORTING NO FIELDS WITH KEY table_line = lv_char.
        IF sy-subrc <> 0.
          APPEND lv_char TO lt_chars.
        ENDIF.
      ENDIF.
    ENDDO.

    rv_count = lines( lt_chars ).
  ENDMETHOD.

ENDCLASS.