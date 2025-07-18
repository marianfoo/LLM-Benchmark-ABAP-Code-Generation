CLASS z_humaneval_118 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: solve
      IMPORTING iv_input         TYPE string
      RETURNING VALUE(rv_result) TYPE string.

  PRIVATE SECTION.

ENDCLASS.

CLASS z_humaneval_118 IMPLEMENTATION.

  METHOD solve.
    DATA: lv_len       TYPE i,
          lv_i         TYPE i,
          lv_char      TYPE c,
          lv_prev_char TYPE c,
          lv_next_char TYPE c.

    lv_len = strlen( iv_input ).

    IF lv_len < 3.
      rv_result = ''.
      RETURN.
    ENDIF.

    lv_i = lv_len - 2.
    WHILE lv_i > 0.
      lv_char = iv_input+lv_i(1).

      IF lv_char = 'a' OR lv_char = 'e' OR lv_char = 'i' OR lv_char = 'o' OR lv_char = 'u' OR
         lv_char = 'A' OR lv_char = 'E' OR lv_char = 'I' OR lv_char = 'O' OR lv_char = 'U'.

        DATA: lv_prev_pos TYPE i,
              lv_next_pos TYPE i.

        lv_prev_pos = lv_i - 1.
        lv_next_pos = lv_i + 1.

        lv_prev_char = iv_input+lv_prev_pos(1).
        lv_next_char = iv_input+lv_next_pos(1).

        IF NOT ( lv_prev_char = 'a' OR lv_prev_char = 'e' OR lv_prev_char = 'i' OR lv_prev_char = 'o' OR lv_prev_char = 'u' OR
                 lv_prev_char = 'A' OR lv_prev_char = 'E' OR lv_prev_char = 'I' OR lv_prev_char = 'O' OR lv_prev_char = 'U' ) AND
           NOT ( lv_next_char = 'a' OR lv_next_char = 'e' OR lv_next_char = 'i' OR lv_next_char = 'o' OR lv_next_char = 'u' OR
                 lv_next_char = 'A' OR lv_next_char = 'E' OR lv_next_char = 'I' OR lv_next_char = 'O' OR lv_next_char = 'U' ).
          rv_result = lv_char.
          RETURN.
        ENDIF.
      ENDIF.

      lv_i = lv_i - 1.
    ENDWHILE.

    rv_result = ''.

  ENDMETHOD.

ENDCLASS.