CLASS z_humaneval_060 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: sum_to_n
      IMPORTING
        n             TYPE i
      RETURNING
        VALUE(result) TYPE i.

ENDCLASS.

CLASS z_humaneval_060 IMPLEMENTATION.

  METHOD sum_to_n.
    DATA: lv_counter TYPE i.

    DO n TIMES.
      lv_counter = sy-index.
      result = result + lv_counter.
    ENDDO.
  ENDMETHOD.

ENDCLASS.