CLASS z_humaneval_007 DEFINITION
PUBLIC
FINAL
ABSTRACT.

  PUBLIC SECTION.
    CLASS-METHODS filter_by_substring
      IMPORTING
        VALUE(strings)          TYPE string_table
        VALUE(substring)        TYPE string
      RETURNING
        VALUE(filtered_strings) TYPE string_table.
ENDCLASS.

CLASS z_humaneval_007 IMPLEMENTATION.
  METHOD filter_by_substring.
    DATA: lv_string TYPE string.
    LOOP AT strings INTO lv_string.
      IF lv_string CP '*' && substring && '*'.
        APPEND lv_string TO filtered_strings.
      ENDIF.
    ENDLOOP.
  ENDMETHOD.
ENDCLASS.