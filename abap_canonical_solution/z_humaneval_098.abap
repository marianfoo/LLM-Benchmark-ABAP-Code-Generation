CLASS z_humaneval_098 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: count_upper
      IMPORTING
        VALUE(s)      TYPE string
      RETURNING
        VALUE(result) TYPE i.

ENDCLASS.

CLASS z_humaneval_098 IMPLEMENTATION.

  METHOD count_upper.
    DATA: lv_count TYPE i VALUE 0,
          lv_len   TYPE i,
          lv_char  TYPE c.

    lv_len = strlen( s ).

    DO lv_len TIMES.
      DATA(lv_index) = sy-index - 1.
      IF lv_index MOD 2 = 0.
        lv_char = s+lv_index(1).
        IF lv_char = 'A' OR lv_char = 'E' OR lv_char = 'I' OR lv_char = 'O' OR lv_char = 'U'.
          lv_count = lv_count + 1.
        ENDIF.
      ENDIF.
    ENDDO.

    result = lv_count.

  ENDMETHOD.

ENDCLASS.