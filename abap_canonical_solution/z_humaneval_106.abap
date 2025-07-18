CLASS z_humaneval_106 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: f
      IMPORTING
        n             TYPE i
      RETURNING
        VALUE(result) TYPE int4_table.

  PRIVATE SECTION.

ENDCLASS.

CLASS z_humaneval_106 IMPLEMENTATION.

  METHOD f.
    DATA: lv_i TYPE i,
          lv_j TYPE i,
          lv_x TYPE i.

    DO n TIMES.
      lv_i = sy-index.

      IF lv_i MOD 2 = 0.
        lv_x = 1.
        DO lv_i TIMES.
          lv_j = sy-index.
          lv_x = lv_x * lv_j.
        ENDDO.
      ELSE.
        lv_x = 0.
        DO lv_i TIMES.
          lv_j = sy-index.
          lv_x = lv_x + lv_j.
        ENDDO.
      ENDIF.

      APPEND lv_x TO result.
    ENDDO.

  ENDMETHOD.

ENDCLASS.