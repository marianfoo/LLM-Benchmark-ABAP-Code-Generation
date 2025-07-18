CLASS z_humaneval_040 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: triples_sum_to_zero
      IMPORTING
        VALUE(l)      TYPE int4_table
      RETURNING
        VALUE(result) TYPE abap_bool.

ENDCLASS.

CLASS z_humaneval_040 IMPLEMENTATION.

  METHOD triples_sum_to_zero.
    DATA: lv_len TYPE i,
          lv_i   TYPE i,
          lv_j   TYPE i,
          lv_k   TYPE i.

    lv_len = lines( l ).

    DO lv_len TIMES.
      lv_i = sy-index.
      lv_j = lv_i + 1.
      WHILE lv_j <= lv_len.
        lv_k = lv_j + 1.
        WHILE lv_k <= lv_len.
          IF l[ lv_i ] + l[ lv_j ] + l[ lv_k ] = 0.
            result = abap_true.
            RETURN.
          ENDIF.
          lv_k = lv_k + 1.
        ENDWHILE.
        lv_j = lv_j + 1.
      ENDWHILE.
    ENDDO.

    result = abap_false.

  ENDMETHOD.

ENDCLASS.