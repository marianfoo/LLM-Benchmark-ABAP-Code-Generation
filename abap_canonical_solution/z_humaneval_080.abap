CLASS z_humaneval_080 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: is_happy
      IMPORTING
        VALUE(s)      TYPE string
      RETURNING
        VALUE(result) TYPE abap_bool.

ENDCLASS.

CLASS z_humaneval_080 IMPLEMENTATION.

  METHOD is_happy.
    DATA: lv_len   TYPE i,
          lv_i     TYPE i,
          lv_pos1  TYPE i,
          lv_pos2  TYPE i,
          lv_pos3  TYPE i,
          lv_char1 TYPE c,
          lv_char2 TYPE c,
          lv_char3 TYPE c.

    lv_len = strlen( s ).

    IF lv_len < 3.
      result = abap_false.
      RETURN.
    ENDIF.

    DO lv_len - 2 TIMES.
      lv_i = sy-index - 1.
      lv_pos1 = lv_i.
      lv_pos2 = lv_i + 1.
      lv_pos3 = lv_i + 2.

      lv_char1 = s+lv_pos1(1).
      lv_char2 = s+lv_pos2(1).
      lv_char3 = s+lv_pos3(1).

      IF lv_char1 = lv_char2 OR
         lv_char2 = lv_char3 OR
         lv_char1 = lv_char3.
        result = abap_false.
        RETURN.
      ENDIF.
    ENDDO.

    result = abap_true.
  ENDMETHOD.

ENDCLASS.